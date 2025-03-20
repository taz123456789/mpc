#define M_PI 3.14159265358979323846
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point.hpp>  // for Point message
#include <visualization_msgs/msg/marker.hpp>
#include <vector>
#include <cmath>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

using namespace std;

struct State {
    double x, y, theta; // position and orientation
    double vx, vy;      // linear velocities in directions x and y 
};

struct Control {//control velocities
    double vx, vy; // linear in x and y directions 
    double omega;  // angular 
};

State predict_state(const State& current_state, const Control& control_input, double dt) {
    State predicted_state = current_state;
    predicted_state.x += control_input.vx * dt;  // x' = x + vx * dt
    predicted_state.y += control_input.vy * dt;  // y' = y + vy * dt
    predicted_state.theta += control_input.omega * dt;  // theta' = theta + omega * dt
    return predicted_state;
}

// cost function = position error + control effort
double cost_function(const State& state, const Control& control, const State& target, double control_weight) {
    double distance = sqrt(pow(state.x - target.x, 2) + pow(state.y - target.y, 2));
    double control_effort = control.vx * control.vx + control.vy * control.vy + control.omega * control.omega;
    double angular_deviation = fabs(atan2(target.y - state.y, target.x - state.x) - state.theta);

    return distance + control_weight * control_effort + 2.0 * angular_deviation;
}

Control nmpc_control(const State& current_state, const State& target, double dt, double control_weight) {
    double learning_rate = 1.0;
    int max_iterations = 100;

    Control control = { 0.05, 0.05, 0.05 }; // initial

    for (int iter = 0; iter < max_iterations; ++iter) {
        State predicted_state = predict_state(current_state, control, dt);

        double cost = cost_function(predicted_state, control, target, control_weight);

        // gradients
        double grad_vx = (cost_function(predict_state(current_state, { control.vx + 0.01, control.vy, control.omega }, dt),
            { control.vx + 0.01, control.vy, control.omega }, target, control_weight) - cost) / 0.01;
        double grad_vy = (cost_function(predict_state(current_state, { control.vx, control.vy + 0.01, control.omega }, dt),
            { control.vx, control.vy + 0.01, control.omega }, target, control_weight) - cost) / 0.01;
        double grad_omega = (cost_function(predict_state(current_state, { control.vx, control.vy, control.omega + 0.01 }, dt),
            { control.vx, control.vy, control.omega + 0.01 }, target, control_weight) - cost) / 0.01;

        //  controls update
        control.vx -= learning_rate * grad_vx;
        control.vy -= learning_rate * grad_vy;
        control.omega -= learning_rate * grad_omega;

        // clamp control inputs
        control.vx = max(-1.0, min(1.0, control.vx));
        control.vy = max(-1.0, min(1.0, control.vy));
        control.omega = max(-M_PI / 4, min(M_PI / 4, control.omega));
    }

    return control;
}

class NMPCControllerNode : public rclcpp::Node {
public:
    NMPCControllerNode()
        : Node("omnidirectional_robot"), current_state_{ 0.0, 0.0, 0.0, 0.0, 0.0 }, dt_(0.05), control_weight_(0.1) {
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 10);
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("path", 10);

        target_position_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/target_position", 10, [this](const geometry_msgs::msg::Point::SharedPtr msg) {
                target_position_ = *msg;
                has_target_ = true;
            });

        timer_ = this->create_wall_timer(
            chrono::milliseconds(static_cast<int>(dt_ * 1000)),
            [this]() { control_loop(); });
    }

private:
    void control_loop() {
        if (!has_target_) {
            RCLCPP_INFO(this->get_logger(), "No target position received yet.");
            return;
        }

        State target = { target_position_.x, target_position_.y, 0.0, 0.0, 0.0 }; // Current target

        // current state and target
        RCLCPP_INFO(this->get_logger(), "Current State: x=%.2f, y=%.2f, theta=%.2f | Target: x=%.2f, y=%.2f",
            current_state_.x, current_state_.y, current_state_.theta, target.x, target.y);

        Control control = nmpc_control(current_state_, target, dt_, control_weight_);

        // state update
        current_state_ = predict_state(current_state_, control, dt_);

        // publish commands
        auto cmd_vel_msg = geometry_msgs::msg::Twist();
        cmd_vel_msg.linear.x = control.vx;
        cmd_vel_msg.linear.y = control.vy;
        cmd_vel_msg.angular.z = control.omega;
        cmd_vel_publisher_->publish(cmd_vel_msg);

        publish_odometry();
        update_path();

        RCLCPP_INFO(this->get_logger(),
            "Control: vx=%.2f, vy=%.2f, omega=%.2f", control.vx, control.vy, control.omega);

        // ensures ifthe target is reached
        double distance_to_target = sqrt(pow(current_state_.x - target.x, 2) + pow(current_state_.y - target.y, 2));
        if (distance_to_target < 0.3) {  // threshold
            RCLCPP_INFO(this->get_logger(), "Target reached.");
            has_target_ = false; // reset target
        }
    }

    void publish_odometry() {
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = now();
        odom_msg.header.frame_id = "odom";
        odom_msg.pose.pose.position.x = current_state_.x;
        odom_msg.pose.pose.position.y = current_state_.y;
        odom_msg.pose.pose.orientation.z = sin(current_state_.theta / 2.0);
        odom_msg.pose.pose.orientation.w = cos(current_state_.theta / 2.0);
        odom_publisher_->publish(odom_msg);
    }

    void update_path() {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = now();
        pose_stamped.header.frame_id = "map";
        pose_stamped.pose.position.x = current_state_.x;
        pose_stamped.pose.position.y = current_state_.y;
        pose_stamped.pose.orientation.z = sin(current_state_.theta / 2.0);
        pose_stamped.pose.orientation.w = cos(current_state_.theta / 2.0);
        path_msg_.poses.push_back(pose_stamped);
        path_publisher_->publish(path_msg_);
    }

    // publishers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_position_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    State current_state_;
    geometry_msgs::msg::Point target_position_;
    nav_msgs::msg::Path path_msg_;
    bool has_target_ = false;
    double dt_;
    double control_weight_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NMPCControllerNode>());
    rclcpp::shutdown();
    return 0;
}
