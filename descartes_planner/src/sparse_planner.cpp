/*
 * Software License Agreement (Apache License)
 *
 * Copyright (c) 2014, Southwest Research Institute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * sparse_planner.cpp
 *
 *  Created on: Dec 17, 2014
 *      Author: Jorge Nicho
 */

#include <descartes_planner/sparse_planner.h>
#include <algorithm>

using namespace descartes_core;
using namespace descartes_trajectory;

namespace
{
descartes_core::TimingConstraint cumulativeTimingBetween(
    const std::vector<descartes_core::TrajectoryPtPtr>& dense_points, size_t start_index, size_t end_index)
{
  descartes_core::TimingConstraint tm(0.0, 0.0);
  for (size_t i = start_index + 1; i <= end_index; ++i)
  {
    const descartes_core::TimingConstraint& pt_tm = dense_points[i]->getTiming();
    if (pt_tm.isSpecified())
    {
      // Add time as normal
      tm.upper += pt_tm.upper;
      tm.lower += pt_tm.lower;
    }
    else
    {
      // A single unspecified timing makes the range unspecified
      tm = descartes_core::TimingConstraint();
      break;
    }
  }
  return tm;
}
}  // namespace

namespace descartes_planner
{
const int INVALID_INDEX = -1;
const double MAX_JOINT_CHANGE = 0.01f;
const double DEFAULT_SAMPLING = 0.01f;
const std::string SAMPLING_CONFIG = "sampling";

SparsePlanner::SparsePlanner(RobotModelConstPtr model, double sampling)
  : sampling_(sampling), error_code_(descartes_core::PlannerError::UNINITIALIZED)
{
  error_map_ = { { PlannerError::OK, "OK" },
                 { PlannerError::EMPTY_PATH, "No path plan has been generated" },
                 { PlannerError::INVALID_ID, "ID is nil or isn't part of the path" },
                 { PlannerError::IK_NOT_AVAILABLE, "One or more ik solutions could not be found" },
                 { PlannerError::UNINITIALIZED, "Planner has not been initialized with a robot model" },
                 { PlannerError::INCOMPLETE_PATH, "Input trajectory and output path point cound differ" } };

  initialize(std::move(model));
  config_ = { { SAMPLING_CONFIG, std::to_string(sampling) } };
}

SparsePlanner::SparsePlanner() : sampling_(DEFAULT_SAMPLING), error_code_(descartes_core::PlannerError::UNINITIALIZED)
{
  error_map_ = { { PlannerError::OK, "OK" },
                 { PlannerError::EMPTY_PATH, "No path plan has been generated" },
                 { PlannerError::INVALID_ID, "ID is nil or isn't part of the path" },
                 { PlannerError::IK_NOT_AVAILABLE, "One or more ik solutions could not be found" },
                 { PlannerError::UNINITIALIZED, "Planner has not been initialized with a robot model" },
                 { PlannerError::INCOMPLETE_PATH, "Input trajectory and output path point cound differ" } };

  config_ = { { SAMPLING_CONFIG, std::to_string(DEFAULT_SAMPLING) } };
}

bool SparsePlanner::initialize(RobotModelConstPtr model)
{
  planning_graph_ =
      boost::shared_ptr<descartes_planner::PlanningGraph>(new descartes_planner::PlanningGraph(std::move(model)));
  error_code_ = PlannerError::EMPTY_PATH;
  return true;
}

bool SparsePlanner::initialize(RobotModelConstPtr model, descartes_planner::CostFunction cost_function_callback)
{
  planning_graph_ = boost::shared_ptr<descartes_planner::PlanningGraph>(
      new descartes_planner::PlanningGraph(std::move(model), cost_function_callback));
  error_code_ = PlannerError::EMPTY_PATH;
  return true;
}

bool SparsePlanner::setConfig(const descartes_core::PlannerConfig& config)
{
  std::stringstream ss;
  static std::vector<std::string> keys = { SAMPLING_CONFIG };

  // verifying keys
  for (auto kv : config_)
  {
    if (config.count(kv.first) == 0)
    {
      error_code_ = descartes_core::PlannerError::INVALID_CONFIGURATION_PARAMETER;
      return false;
    }
  }

  // translating string values
  try
  {
    config_[SAMPLING_CONFIG] = config.at(SAMPLING_CONFIG);
    sampling_ = std::stod(config.at(SAMPLING_CONFIG));
  }
  catch (std::invalid_argument& exp)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Unable to parse configuration value(s)");
    error_code_ = descartes_core::PlannerError::INVALID_CONFIGURATION_PARAMETER;
    return false;
  }

  return true;
}

void SparsePlanner::getConfig(descartes_core::PlannerConfig& config) const
{
  config = config_;
}

SparsePlanner::~SparsePlanner()
{
}

void SparsePlanner::setSampling(double sampling)
{
  sampling_ = sampling;
}

bool SparsePlanner::planPath(const std::vector<TrajectoryPtPtr>& traj)
{
  if (error_code_ == descartes_core::PlannerError::UNINITIALIZED)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Planner has not been initialized");
    return false;
  }

  rclcpp::Clock clock = rclcpp::Clock{};
  rclcpp::Time start_time =  clock.now();

  cart_points_.assign(traj.begin(), traj.end());
  std::vector<TrajectoryPtPtr> sparse_trajectory_array;
  sampleTrajectory(sampling_, cart_points_, sparse_trajectory_array);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("descartes_planner"),"Sampled trajectory contains " << sparse_trajectory_array.size() << " points from "
                                                 << cart_points_.size() << " points in the dense trajectory");

  if (planning_graph_->insertGraph(sparse_trajectory_array) && plan())
  {
    int planned_count = sparse_solution_array_.size();
    int interp_count = cart_points_.size() - sparse_solution_array_.size();
    RCLCPP_INFO(rclcpp::get_logger("descartes_planner"),"Sparse planner succeeded with %i planned point and %i interpolated points in %f seconds", planned_count,
             interp_count, (clock.now() - start_time).seconds());
    error_code_ = descartes_core::PlannerError::OK;
  }
  else
  {
    error_code_ = descartes_core::PlannerError::IK_NOT_AVAILABLE;
    return false;
  }

  return true;
}

bool SparsePlanner::addAfter(const TrajectoryPt::ID& ref_id, TrajectoryPtPtr cp)
{
  rclcpp::Clock clock = rclcpp::Clock{};
  rclcpp::Time start_time = clock.now();
  int sparse_index;
  int index;
  TrajectoryPt::ID prev_id, next_id;

  sparse_index = findNearestSparsePointIndex(ref_id);
  if (sparse_index == INVALID_INDEX)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"A point in sparse array near point " << ref_id << " could not be found, aborting");
    return false;
  }

  // setting ids from sparse array
  prev_id = std::get<1>(sparse_solution_array_[sparse_index - 1])->getID();
  next_id = std::get<1>(sparse_solution_array_[sparse_index])->getID();

  // inserting into dense array
  index = getDensePointIndex(ref_id);
  if (index == INVALID_INDEX)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Point  " << ref_id << " could not be found in dense array, aborting");
    return false;
  }
  auto pos = cart_points_.begin();
  std::advance(pos, index + 1);
  cart_points_.insert(pos, cp);

  // replanning
  if (planning_graph_->addTrajectory(cp, prev_id, next_id) && plan())
  {
    int planned_count = sparse_solution_array_.size();
    int interp_count = cart_points_.size() - sparse_solution_array_.size();
    RCLCPP_INFO(rclcpp::get_logger("descartes_planner"),"Sparse planner add operation succeeded, %i planned point and %i interpolated points in %f seconds",
             planned_count, interp_count, (clock.now() - start_time).seconds());
  }
  else
  {
    return false;
  }

  return true;
}

bool SparsePlanner::addBefore(const TrajectoryPt::ID& ref_id, TrajectoryPtPtr cp)
{
  rclcpp::Clock clock = rclcpp::Clock{};
  rclcpp::Time start_time = clock.now();
  int sparse_index;
  int index;
  TrajectoryPt::ID prev_id, next_id;

  sparse_index = findNearestSparsePointIndex(ref_id, false);
  if (sparse_index == INVALID_INDEX)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"A point in sparse array near point " << ref_id << " could not be found, aborting");
    return false;
  }

  prev_id = (sparse_index == 0) ? descartes_core::TrajectoryID::make_nil() :
                                  std::get<1>(sparse_solution_array_[sparse_index - 1])->getID();
  next_id = std::get<1>(sparse_solution_array_[sparse_index])->getID();

  // inserting into dense array
  index = getDensePointIndex(ref_id);
  if (index == INVALID_INDEX)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Point  " << ref_id << " could not be found in dense array, aborting");
    return false;
  }
  auto pos = cart_points_.begin();
  std::advance(pos, index);
  cart_points_.insert(pos, cp);

  if (planning_graph_->addTrajectory(cp, prev_id, next_id) && plan())
  {
    int planned_count = sparse_solution_array_.size();
    int interp_count = cart_points_.size() - sparse_solution_array_.size();
    RCLCPP_INFO(rclcpp::get_logger("descartes_planner"),"Sparse planner add operation succeeded, %i planned point and %i interpolated points in %f seconds",
             planned_count, interp_count, (clock.now() - start_time).seconds());
  }
  else
  {
    return false;
  }

  return true;
}

bool SparsePlanner::remove(const TrajectoryPt::ID& ref_id)
{
  rclcpp::Clock clock = rclcpp::Clock{};
  rclcpp::Time start_time = clock.now();
  int index = getDensePointIndex(ref_id);
  if (index == INVALID_INDEX)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Point  " << ref_id << " could not be found in dense array, aborting");
    return false;
  }

  if (isInSparseTrajectory(ref_id))
  {
    if (!planning_graph_->removeTrajectory(cart_points_[index]->getID()))
    {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Failed to removed point " << ref_id << " from sparse trajectory, aborting");
      return false;
    }
  }

  // removing from dense array
  auto pos = cart_points_.begin();
  std::advance(pos, index);
  cart_points_.erase(pos);

  if (plan())
  {
    int planned_count = sparse_solution_array_.size();
    int interp_count = cart_points_.size() - sparse_solution_array_.size();
    RCLCPP_INFO(rclcpp::get_logger("descartes_planner"),"Sparse planner remove operation succeeded, %i planned point and %i interpolated points in %f seconds",
             planned_count, interp_count, (clock.now() - start_time).seconds());
  }
  else
  {
    return false;
  }

  return true;
}

bool SparsePlanner::modify(const TrajectoryPt::ID& ref_id, TrajectoryPtPtr cp)
{
  rclcpp::Clock clock = rclcpp::Clock{};
  rclcpp::Time start_time = clock.now();
  int sparse_index;
  TrajectoryPt::ID prev_id, next_id;

  sparse_index = getSparsePointIndex(ref_id);
  cp->setID(ref_id);
  if (sparse_index == INVALID_INDEX)
  {
    sparse_index = findNearestSparsePointIndex(ref_id);
    prev_id = std::get<1>(sparse_solution_array_[sparse_index - 1])->getID();
    next_id = std::get<1>(sparse_solution_array_[sparse_index])->getID();
    if (!planning_graph_->addTrajectory(cp, prev_id, next_id))
    {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Failed to add point to sparse trajectory, aborting");
      return false;
    }
  }
  else
  {
    if (!planning_graph_->modifyTrajectory(cp))
    {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Failed to modify point in sparse trajectory, aborting");
      return false;
    }
  }

  int index = getDensePointIndex(ref_id);
  cart_points_[index] = cp;
  if (plan())
  {
    int planned_count = sparse_solution_array_.size();
    int interp_count = cart_points_.size() - sparse_solution_array_.size();
    RCLCPP_INFO(rclcpp::get_logger("descartes_planner"),"Sparse planner modify operation succeeded, %i planned point and %i interpolated points in %f seconds",
             planned_count, interp_count, (clock.now() - start_time).seconds());
  }
  else
  {
    return false;
  }

  return true;
}

bool SparsePlanner::isInSparseTrajectory(const TrajectoryPt::ID& ref_id)
{
  auto predicate = [&ref_id](std::tuple<int, TrajectoryPtPtr, JointTrajectoryPt>& t) {
    return ref_id == std::get<1>(t)->getID();
  };

  return (std::find_if(sparse_solution_array_.begin(), sparse_solution_array_.end(), predicate) !=
          sparse_solution_array_.end());
}

int SparsePlanner::getDensePointIndex(const TrajectoryPt::ID& ref_id)
{
  int index = INVALID_INDEX;
  auto predicate = [&ref_id](TrajectoryPtPtr cp) { return ref_id == cp->getID(); };

  auto pos = std::find_if(cart_points_.begin(), cart_points_.end(), predicate);
  if (pos != cart_points_.end())
  {
    index = std::distance(cart_points_.begin(), pos);
  }

  return index;
}

int SparsePlanner::getSparsePointIndex(const TrajectoryPt::ID& ref_id)
{
  int index = INVALID_INDEX;
  auto predicate = [ref_id](std::tuple<int, TrajectoryPtPtr, JointTrajectoryPt>& t) {
    return ref_id == std::get<1>(t)->getID();
  };

  auto pos = std::find_if(sparse_solution_array_.begin(), sparse_solution_array_.end(), predicate);
  if (pos != sparse_solution_array_.end())
  {
    index = std::distance(sparse_solution_array_.begin(), pos);
  }

  return index;
}

int SparsePlanner::findNearestSparsePointIndex(const TrajectoryPt::ID& ref_id, bool skip_equal)
{
  int index = INVALID_INDEX;
  int dense_index = getDensePointIndex(ref_id);

  if (dense_index == INVALID_INDEX)
  {
    return index;
  }

  auto predicate = [&dense_index, &skip_equal](std::tuple<int, TrajectoryPtPtr, JointTrajectoryPt>& t) {
    if (skip_equal)
    {
      return dense_index < std::get<0>(t);
    }
    else
    {
      return dense_index <= std::get<0>(t);
    }
  };

  auto pos = std::find_if(sparse_solution_array_.begin(), sparse_solution_array_.end(), predicate);
  if (pos != sparse_solution_array_.end())
  {
    index = std::distance(sparse_solution_array_.begin(), pos);
  }
  else
  {
    index = sparse_solution_array_.size() - 1;  // last
  }

  return index;
}

bool SparsePlanner::getSparseSolutionArray(SolutionArray& sparse_solution_array)
{
  std::list<JointTrajectoryPt> sparse_joint_points;
  std::vector<TrajectoryPtPtr> sparse_cart_points;
  double cost;
  rclcpp::Clock clock = rclcpp::Clock{};
  rclcpp::Time start_time = clock.now();
  if (planning_graph_->getShortestPath(cost, sparse_joint_points))
  {
    RCLCPP_INFO_STREAM(rclcpp::get_logger("descartes_planner"),"Sparse solution was found in " << (clock.now() - start_time).seconds() << " seconds");
    bool success =
        getOrderedSparseArray(sparse_cart_points) && (sparse_joint_points.size() == sparse_cart_points.size());
    if (!success)
    {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Failed to retrieve sparse solution due to unequal array sizes, cartetian pts: "
                       << sparse_cart_points.size() << ", joints pts: " << sparse_joint_points.size());
      return false;
    }
  }
  else
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Failed to find sparse joint solution");
    return false;
  }

  unsigned int i = 0;
  unsigned int index;
  sparse_solution_array.clear();
  sparse_solution_array.reserve(sparse_cart_points.size());
  for (auto& item : sparse_joint_points)
  {
    TrajectoryPtPtr cp = sparse_cart_points[i++];
    JointTrajectoryPt& jp = item;
    index = getDensePointIndex(cp->getID());

    if (index == INVALID_INDEX)
    {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Cartesian point " << cp->getID() << " not found");
      return false;
    }
    else
    {
      RCLCPP_DEBUG_STREAM(rclcpp::get_logger("descartes_planner"),"Point with dense index " << index << " and id " << cp->getID() << " added to sparse");
    }

    sparse_solution_array.push_back(std::make_tuple(index, cp, jp));
  }
  return true;
}

bool SparsePlanner::getOrderedSparseArray(std::vector<TrajectoryPtPtr>& sparse_array)
{
  // 1. Find the first point in the original trajectory
  // 2. Copy point pointers from this to into 'sparse_array' in sequence
  const auto& graph = planning_graph_->graph();
  sparse_array.resize(graph.size());
  for (std::size_t i = 0; i < graph.size(); ++i)
  {
    auto idx = getDensePointIndex(graph.getRung(i).id);
    sparse_array[i] = cart_points_[idx];
  }
  return true;
}

bool SparsePlanner::getSolutionJointPoint(const CartTrajectoryPt::ID& cart_id, JointTrajectoryPt& j)
{
  if (joint_points_map_.count(cart_id) > 0)
  {
    j = joint_points_map_[cart_id];
  }
  else
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Solution for point " << cart_id << " was not found");
    return false;
  }

  return true;
}

bool SparsePlanner::getPath(std::vector<TrajectoryPtPtr>& path) const
{
  if (cart_points_.empty() || joint_points_map_.empty())
  {
    return false;
  }

  path.resize(cart_points_.size());
  for (int i = 0; i < cart_points_.size(); i++)
  {
    TrajectoryPtPtr p = cart_points_[i];
    const JointTrajectoryPt& j = joint_points_map_.at(p->getID());
    TrajectoryPtPtr new_pt = TrajectoryPtPtr(new JointTrajectoryPt(j));
    path[i] = new_pt;
  }

  return true;
}

int SparsePlanner::getErrorCode() const
{
  return error_code_;
}

bool SparsePlanner::getErrorMessage(int error_code, std::string& msg) const
{
  std::map<int, std::string>::const_iterator it = error_map_.find(error_code);

  if (it != error_map_.cend())
  {
    msg = it->second;
  }
  else
  {
    return false;
  }
  return true;
}

void SparsePlanner::sampleTrajectory(double sampling, const std::vector<TrajectoryPtPtr>& dense_trajectory_array,
                                     std::vector<TrajectoryPtPtr>& sparse_trajectory_array)
{
  std::stringstream ss;
  int skip = std::ceil(double(1.0f) / sampling);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("descartes_planner"),"Sampling skip val: " << skip << " from sampling val: " << sampling);
  ss << "[";

  if (dense_trajectory_array.empty())
    return;

  // Add the first point
  sparse_trajectory_array.push_back(dense_trajectory_array.front());
  ss << "0 ";
  // The first point requires no special timing adjustment

  int i;  // We keep i outside of the loop so we can examine it on the last step
  for (i = skip; i < dense_trajectory_array.size(); i += skip)
  {
    // Add the cumulative time of the dense trajectory back in
    descartes_core::TimingConstraint tm = cumulativeTimingBetween(dense_trajectory_array, i - skip, i);
    // We don't want to modify the input trajectory pointers, so we clone and modify them here
    descartes_core::TrajectoryPtPtr cloned = dense_trajectory_array[i]->copyAndSetTiming(tm);
    // Write to the new array
    sparse_trajectory_array.push_back(cloned);

    ss << i << " ";
  }

  // add the last one
  if (sparse_trajectory_array.back()->getID() != dense_trajectory_array.back()->getID())
  {
    // The final point is index size() - 1
    // The point before is index ( i - skip )
    descartes_core::TimingConstraint tm =
        cumulativeTimingBetween(dense_trajectory_array, i - skip, dense_trajectory_array.size() - 1);
    // We don't want to modify the input trajectory pointers, so we clone and modify them here
    descartes_core::TrajectoryPtPtr cloned = dense_trajectory_array.back()->copyAndSetTiming(tm);
    // Write to solution
    sparse_trajectory_array.push_back(cloned);
    ss << dense_trajectory_array.size() - 1 << " ";
  }
  ss << "]";
  RCLCPP_INFO_STREAM(rclcpp::get_logger("descartes_planner"),"Sparse Indices: " << ss.str());
}

bool SparsePlanner::interpolateJointPose(const std::vector<double>& start, const std::vector<double>& end, double t,
                                         std::vector<double>& interp)
{
  if (start.size() != end.size())
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Joint arrays have unequal size, interpolation failed");
    return false;
  }

  if ((t > 1 || t < 0))
  {
    return false;
  }

  interp.resize(start.size());
  double val = 0.0f;
  for (int i = 0; i < start.size(); i++)
  {
    val = end[i] - (end[i] - start[i]) * (1 - t);
    interp[i] = val;
  }

  return true;
}

bool SparsePlanner::plan()
{
  // solving coarse trajectory
  bool replan = true;
  bool succeeded = false;
  int replanning_attempts = 0;
  while (replan && getSparseSolutionArray(sparse_solution_array_))
  {
    // sparse_index is the index in the sampled trajectory that a new point is to be added
    // point_pos is the index into the dense trajectory that the new point is to be copied from
    int sparse_index, point_pos;
    int result = interpolateSparseTrajectory(sparse_solution_array_, sparse_index, point_pos);
    TrajectoryPt::ID prev_id, next_id;
    TrajectoryPtPtr cart_point;
    switch (result)
    {
      case int(InterpolationResult::REPLAN):
        replan = true;
        cart_point = cart_points_[point_pos];
        if (sparse_index == 0)
        {
          // If the point is being inserted at the beginning of the trajectory
          // there is no need to tweak the timing that comes from the dense traj
          prev_id = descartes_core::TrajectoryID::make_nil();
          next_id = std::get<1>(sparse_solution_array_[sparse_index])->getID();
        }
        else
        {
          // Here we want to calculate the time from the prev point to the new point
          int prev_dense_id = std::get<0>(sparse_solution_array_[sparse_index - 1]);
          int next_dense_id = point_pos;
          descartes_core::TimingConstraint tm = cumulativeTimingBetween(cart_points_, prev_dense_id, next_dense_id);
          descartes_core::TrajectoryPtPtr copy_pt = cart_point->copyAndSetTiming(tm);
          cart_point = copy_pt;  // swap the point over

          prev_id = std::get<1>(sparse_solution_array_[sparse_index - 1])->getID();
          next_id = std::get<1>(sparse_solution_array_[sparse_index])->getID();
        }

        // In either case, the sparse_index point will have to be recalculated
        {
          int prev_dense_id = point_pos;
          int next_dense_id = std::get<0>(sparse_solution_array_[sparse_index]);
          descartes_core::TimingConstraint tm = cumulativeTimingBetween(cart_points_, prev_dense_id, next_dense_id);
          descartes_core::TrajectoryPtPtr copy_pt = cart_points_[next_dense_id]->copyAndSetTiming(tm);

          if (!planning_graph_->modifyTrajectory(copy_pt))
          {
            // Theoretically, this should never occur as we are merely modifying an existing point in the sparse
            // graph.
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Could not modify trajectory point with id: " << copy_pt->getID());
            replan = false;
            succeeded = false;
            break;
          }

          // Add into original trajectory
          if (planning_graph_->addTrajectory(cart_point, prev_id, next_id))
          {
            sparse_solution_array_.clear();
            RCLCPP_INFO_STREAM(rclcpp::get_logger("descartes_planner"),"Added new point to sparse trajectory from dense trajectory at position "
                            << point_pos << ", re-planning entire trajectory");
          }
          else
          {
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Adding point " << point_pos << "to sparse trajectory failed, aborting");
            replan = false;
            succeeded = false;
          }
        }

        break;
      case int(InterpolationResult::SUCCESS):
        replan = false;
        succeeded = true;
        break;
      case int(InterpolationResult::ERROR):
        replan = false;
        succeeded = false;
        break;
    }
  }

  return succeeded;
}

double SparsePlanner::maxJointChange(const std::vector<double>& s1, const std::vector<double>& s2)
{
  if (s1.size() != s2.size())
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Joint arrays have unequal size, failed to check for large joint changes");
    return 1000;
  }

  double max_change = 0;
  for (int i = 0; i < s1.size(); i++)
  {
    int change = std::abs(s1[i] - s2[i]);
    if (change > max_change)
      max_change = change;
  }
  return max_change;
}

int SparsePlanner::interpolateSparseTrajectory(const SolutionArray& sparse_solution_array, int& sparse_index,
                                               int& point_pos)
{
  // populating full path
  joint_points_map_.clear();
  descartes_core::RobotModelConstPtr robot_model = planning_graph_->getRobotModel();
  std::vector<double> start_jpose, end_jpose, rough_interp, aprox_interp, seed_pose(robot_model->getDOF(), 0);
  for (int k = 1; k < sparse_solution_array.size(); k++)
  {
    auto start_index = std::get<0>(sparse_solution_array[k - 1]);
    auto end_index = std::get<0>(sparse_solution_array[k]);
    TrajectoryPtPtr start_tpoint = std::get<1>(sparse_solution_array[k - 1]);
    TrajectoryPtPtr end_tpoint = std::get<1>(sparse_solution_array[k]);
    const JointTrajectoryPt& start_jpoint = std::get<2>(sparse_solution_array[k - 1]);
    const JointTrajectoryPt& end_jpoint = std::get<2>(sparse_solution_array[k]);

    start_jpoint.getNominalJointPose(seed_pose, *robot_model, start_jpose);
    end_jpoint.getNominalJointPose(seed_pose, *robot_model, end_jpose);

    // adding start joint point to solution
    joint_points_map_.insert(std::make_pair(start_tpoint->getID(), start_jpoint));

    // interpolating
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("descartes_planner"),"Interpolation parameters: start index " << start_index << ", end index " << end_index);

    double max_joint_change_found = 0;
    int max_joint_change_index = 0;
    for (int pos = start_index + 1; (pos <= end_index) && (pos < cart_points_.size()); pos++)
    {
      double t = double(pos - start_index) / double(end_index - start_index);
      if (!interpolateJointPose(start_jpose, end_jpose, t, rough_interp))
      {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("descartes_planner"),"Interpolation for point at position " << pos << "failed, aborting");
        return (int)InterpolationResult::ERROR;
      }

      TrajectoryPtPtr cart_point = cart_points_[pos];
      if (pos != end_index)
      {
        if (cart_point->getClosestJointPose(rough_interp, *robot_model, aprox_interp))
        {
          double joint_change = maxJointChange(rough_interp, aprox_interp);
          if (joint_change > max_joint_change_found)
          {
            max_joint_change_found = joint_change;
            max_joint_change_index = pos;
          }
          RCLCPP_DEBUG_STREAM(rclcpp::get_logger("descartes_planner"),"Interpolated point at position " << pos);

          // look up previous points joint solution
          const JointTrajectoryPt& last_joint_pt = joint_points_map_.at(cart_points_[pos - 1]->getID());
          std::vector<double> last_joint_pose;
          last_joint_pt.getNominalJointPose(std::vector<double>(), *robot_model, last_joint_pose);

          // retreiving timing constraint
          // TODO, let's check the timing constraints
          const descartes_core::TimingConstraint& tm = cart_points_[pos]->getTiming();

          // check validity of joint motion
          if (tm.isSpecified() && !robot_model->isValidMove(last_joint_pose, aprox_interp, tm.upper))
          {
            RCLCPP_WARN_STREAM(rclcpp::get_logger("descartes_planner"),"Joint velocity checking failed for point " << pos << ". Replanning.");
            point_pos = pos;
            sparse_index = k;
            return static_cast<int>(InterpolationResult::REPLAN);
          }

          joint_points_map_.insert(std::make_pair(cart_point->getID(), JointTrajectoryPt(aprox_interp, tm)));
        }
        else
        {
          RCLCPP_WARN_STREAM(rclcpp::get_logger("descartes_planner"),"Couldn't find a closest joint pose for point " << cart_point->getID() << ", replanning");
          sparse_index = k;
          point_pos = pos;
          return (int)InterpolationResult::REPLAN;
        }
      }
      else  // j == step
      {
        const JointTrajectoryPt& last_joint_pt = joint_points_map_.at(cart_points_[pos - 1]->getID());
        std::vector<double> last_joint_pose;
        last_joint_pt.getNominalJointPose(std::vector<double>(), *robot_model, last_joint_pose);
        const descartes_core::TimingConstraint& tm = cart_points_[pos]->getTiming();

        if (tm.isSpecified() && !robot_model->isValidMove(last_joint_pose, rough_interp, tm.upper))
        {
          RCLCPP_WARN_STREAM(rclcpp::get_logger("descartes_planner"),"Joint velocity checking failed for last-point " << pos << ". Adding previous point.");
          point_pos = (pos - 1);
          sparse_index = k;
          return static_cast<int>(InterpolationResult::REPLAN);
        }
        joint_points_map_.insert(std::make_pair(cart_point->getID(), JointTrajectoryPt(rough_interp, tm)));
      }
    }  // end of interpolation steps between sparse points

    if (max_joint_change_found > MAX_JOINT_CHANGE)
    {
      RCLCPP_WARN_STREAM(rclcpp::get_logger("descartes_planner"),"Joint changes greater that " << MAX_JOINT_CHANGE << " detected for point "
                                                    << max_joint_change_index << ", replanning");
      sparse_index = k;
      point_pos = max_joint_change_index;
      return (int)InterpolationResult::REPLAN;
    }
  }  // end of sparse point loop

  return (int)InterpolationResult::SUCCESS;
}

} /* namespace descartes_planner */
