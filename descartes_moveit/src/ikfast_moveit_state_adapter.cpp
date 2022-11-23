/*
 * Software License Agreement (Apache License)
 *
 * Copyright (c) 2016, Jonathan Meyer
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

#include "descartes_moveit/ikfast_moveit_state_adapter.h"

#include <eigen_conversions/eigen_msg.h>
#include <ros/node_handle.h>

const static std::string DEFAULT_BASE_FRAME = "base_link";
const static std::string DEFAULT_TOOL_FRAME = "tool0";

// Compute the 'joint distance' between two poses
static double distance(const std::vector<double>& a, const std::vector<double>& b)
{
  double cost = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    cost += std::abs(b[i] - a[i]);
  return cost;
}

// Compute the index of the closest joint pose in 'candidates' from 'target'
static size_t closestJointPose(const std::vector<double>& target, const std::vector<std::vector<double>>& candidates)
{
  size_t closest = 0;  // index into candidates
  double lowest_cost = std::numeric_limits<double>::max();
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    assert(target.size() == candidates[i].size());
    double c = distance(target, candidates[i]);
    if (c < lowest_cost)
    {
      closest = i;
      lowest_cost = c;
    }
  }
  return closest;
}

bool descartes_moveit::IkFastMoveitStateAdapter::initialize(const std::string& robot_description,
                                                            const std::string& group_name,
                                                            const std::string& world_frame,
                                                            const std::string& tcp_frame)
{
  if (!MoveitStateAdapter::initialize(robot_description, group_name, world_frame, tcp_frame))
  {
    return false;
  }

  return computeIKFastTransforms(world_frame, tcp_frame);
}

bool descartes_moveit::IkFastMoveitStateAdapter::initialize(planning_scene_monitor::PlanningSceneMonitorPtr& psm,
                                                            const std::string& group_name,
                                                            const std::string& world_frame,
                                                            const std::string& tcp_frame)
{
  if (!MoveitStateAdapter::initialize(psm, group_name, world_frame, tcp_frame))
  {
    return false;
  }

  return computeIKFastTransforms(world_frame, tcp_frame);
}

// inspired by https://stackoverflow.com/a/48271759
// Returns all possible lists where
// list[i] ∈ options[i] ∀i
std::vector<std::vector<double>> combinations(const std::vector<std::vector<double>>& options)
{
  size_t nComb = 1;
  size_t N = options.size();

  for (const auto& v : options)
    nComb *= v.size();

  std::vector<std::vector<double>> res(nComb, std::vector<double>(N));

  for (size_t i = 0; i < nComb; i++)
  {
    auto temp = i;
    for (size_t j = 0; j < N; j++)
    {
      auto index = temp % options[j].size();
      temp /= options[j].size();
      res[i][j]=options[j][index];
    };
  }
  return res;
}

// For a single joint, get all of its variants
// For a joint that goes from -360 degrees to 360 degrees:
// the variants of pose 90 are [-270, 90]
// the variants of pose -90 are [-90, 270]
std::vector<double> variantsOfJoint(double pose, moveit::core::VariableBounds bounds)
{
  std::vector<double> res{ pose };
  for (double variant = pose - 2 * M_PI; bounds.min_position_ <= variant; variant -= 2 * M_PI)
    res.push_back(variant);

  for (double variant = pose + 2 * M_PI; variant <= bounds.max_position_; variant += 2 * M_PI)
    res.push_back(variant);

  return res;
}

// For a single ik solution, get all variants by combining the variants of all poses
std::vector<std::vector<double>>
descartes_moveit::IkFastMoveitStateAdapter::variantsOfIK(const std::vector<double>& joint_pose) const
{
  std::vector<const std::vector<moveit::core::VariableBounds>*> bounds = joint_group_->getActiveJointModelsBounds();

  std::vector<std::vector<double>> jointVariants;  // jointVariants[i] is a list of all variants of joint i
  for (uint i = 0; i < joint_pose.size(); i++)
    jointVariants.push_back(variantsOfJoint(joint_pose[i], (*bounds[i])[0]));

  return combinations(jointVariants);
}

bool descartes_moveit::IkFastMoveitStateAdapter::getAllIK(const Eigen::Isometry3d& pose,
                                                          std::vector<std::vector<double>>& joint_poses) const
{
  joint_poses.clear();
  const auto& solver = joint_group_->getSolverInstance();

  // Transform input pose
  Eigen::Isometry3d tool_pose = world_to_base_.frame_inv * pose * tool0_to_tip_.frame;

  // convert to geometry_msgs ...
  geometry_msgs::Pose geometry_pose;
  tf::poseEigenToMsg(tool_pose, geometry_pose);
  std::vector<geometry_msgs::Pose> poses = { geometry_pose };

  std::vector<double> dummy_seed(getDOF(), 0.0);
  std::vector<std::vector<double>> joint_results;
  kinematics::KinematicsResult result;
  kinematics::KinematicsQueryOptions options;  // defaults are reasonable as of Indigo

  if (!solver->getPositionIK(poses, dummy_seed, joint_results, result, options))
  {
    return false;
  }

  for (auto& sol : joint_results)
    if (isValid(sol))
      for (auto& variant : variantsOfIK(sol))
        joint_poses.push_back(std::move(variant));

  return joint_poses.size() > 0;
}

bool descartes_moveit::IkFastMoveitStateAdapter::getIK(const Eigen::Isometry3d& pose,
                                                       const std::vector<double>& seed_state,
                                                       std::vector<double>& joint_pose) const
{
  // Descartes Robot Model interface calls for 'closest' point to seed position
  std::vector<std::vector<double>> joint_poses;
  if (!getAllIK(pose, joint_poses))
    return false;
  // Find closest joint pose; getAllIK() does isValid checks already
  joint_pose = joint_poses[closestJointPose(seed_state, joint_poses)];
  return true;
}

bool descartes_moveit::IkFastMoveitStateAdapter::getFK(const std::vector<double>& joint_pose,
                                                       Eigen::Isometry3d& pose) const
{
  const auto& solver = joint_group_->getSolverInstance();

  std::vector<std::string> tip_frame = { solver->getTipFrame() };
  std::vector<geometry_msgs::Pose> output;

  if (!isValid(joint_pose))
    return false;

  if (!solver->getPositionFK(tip_frame, joint_pose, output))
    return false;

  tf::poseMsgToEigen(output[0], pose);  // pose in frame of IkFast base
  pose = world_to_base_.frame * pose * tool0_to_tip_.frame_inv;
  return true;
}

void descartes_moveit::IkFastMoveitStateAdapter::setState(const moveit::core::RobotState& state)
{
  descartes_moveit::MoveitStateAdapter::setState(state);
  computeIKFastTransforms();
}

bool descartes_moveit::IkFastMoveitStateAdapter::computeIKFastTransforms()
{
  return computeIKFastTransforms(DEFAULT_BASE_FRAME, DEFAULT_TOOL_FRAME);
}

bool descartes_moveit::IkFastMoveitStateAdapter::computeIKFastTransforms(std::string base_frame, std::string tool_frame)
{
  // look up the IKFast base and tool frame
  ros::NodeHandle nh;
  std::string ikfast_base_frame, ikfast_tool_frame;
  nh.param<std::string>("ikfast_base_frame", ikfast_base_frame, base_frame);
  nh.param<std::string>("ikfast_tool_frame", ikfast_tool_frame, tool_frame);

  if (!robot_state_->knowsFrameTransform(ikfast_base_frame))
  {
    CONSOLE_BRIDGE_logError("IkFastMoveitStateAdapter: Cannot find transformation to frame '%s' in group '%s'.",
                            ikfast_base_frame.c_str(), group_name_.c_str());
    return false;
  }

  if (!robot_state_->knowsFrameTransform(ikfast_tool_frame))
  {
    CONSOLE_BRIDGE_logError("IkFastMoveitStateAdapter: Cannot find transformation to frame '%s' in group '%s'.",
                            ikfast_tool_frame.c_str(), group_name_.c_str());
    return false;
  }

  // calculate frames
  tool0_to_tip_ = descartes_core::Frame(robot_state_->getFrameTransform(tool_frame_).inverse() *
                                        robot_state_->getFrameTransform(ikfast_tool_frame));

  world_to_base_ = descartes_core::Frame(world_to_root_.frame * robot_state_->getFrameTransform(ikfast_base_frame));

  CONSOLE_BRIDGE_logInform("IkFastMoveitStateAdapter: initialized with IKFast tool frame '%s' and base frame '%s'.",
                           ikfast_tool_frame.c_str(), ikfast_base_frame.c_str());
  return true;
}
