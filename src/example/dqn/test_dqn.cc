#include <stdio.h>
#include <string>
#include <cassert>
#include <iterator>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "include/gym/gym.h"
#include "src/function.h"
#include "src/optimizer.h"


struct Item {
    std::vector<float> state;
    std::vector<float> action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class ReplayBuffer {
    std::vector<Item> ring_;
    int start_;
    int size_;

public:
    ReplayBuffer(int size) : ring_(size), start_(0), size_(0) {}
    
    void add(Item item) {
        ring_[start_] = item;
        start_++;
        if (start_ >= ring_.size()) {
            start_ = 0;
        }
        
        if (size_ < ring_.size()) {
            size_++;
        }
    }
    
    int size() {
        return size_;
    }

    Item sample() {
        return ring_[rand() % size_];
    }
};

class DQN {
public:
    DQN(int state_count, int action_count) : state_count_(state_count),
                                             action_count_(action_count),
                                             linear1_(state_count, 20, "tanh"),
                                             linear2_(20, action_count),
                                             adam_(0.01) {
        auto cpu_place = new paddle::platform::CPUPlace();
        paddle::platform::DeviceContextPool &pool = paddle::platform::DeviceContextPool::Instance();
        dev_ctx_ = pool.Get(*cpu_place);
    }

    void train_step(std::vector<Item> items) {
        paddle::tape::reset_global_tape();

        auto forward = [&](paddle::tape::VariableHandle input) -> paddle::tape::VariableHandle {
            return linear2_(linear1_(input));
        };

        std::vector<float> states;
        for (auto item : items) {
            states.insert(std::end(states), std::begin(item.state), std::end(item.state));
        }

        auto input = paddle::tape::VariableHandle(new paddle::tape::Variable("state"));
        auto tensor = input->GetMutable<paddle::framework::Tensor>();
        fill_state_tensor(*dev_ctx_, states, tensor, items.size());
        auto action_values = forward(input);
        auto action_values_tensor = action_values->Get<paddle::framework::Tensor>();

        std::vector<float> next_states;
        for (auto item : items) {
            next_states.insert(std::end(next_states), std::begin(item.state), std::end(item.state));
        }

        auto next_input = paddle::tape::VariableHandle(new paddle::tape::Variable("next_state"));
        auto next_tensor = next_input->GetMutable<paddle::framework::Tensor>();
        fill_state_tensor(*dev_ctx_, next_states, next_tensor, items.size());
        auto next_state_action_values = forward(next_input);
        auto next_state_action_tensor = next_state_action_values->Get<paddle::framework::Tensor>();
        std::vector<float> next_state_action_vec;
        paddle::framework::TensorToVector(next_state_action_tensor, *dev_ctx_, &next_state_action_vec);

        // calculate the label
        std::vector<float> target_state_action_values;
        paddle::framework::TensorToVector(action_values_tensor, *dev_ctx_, &target_state_action_values);
        for (int i = 0; i < items.size(); i++) {
            auto item = items[i];
            auto idx = i * action_count_ + int(item.action[0]);

            if (item.done) {
                target_state_action_values[idx] = item.reward;
            } else {
                float max_next_state_value;
                for (int j = i * action_count_; j < (i+1)*action_count_; j++) {
                    if (j == i * action_count_) {
                        max_next_state_value = next_state_action_vec[j];
                        continue;
                    }

                    if (max_next_state_value < next_state_action_vec[j]) {
                        max_next_state_value = next_state_action_vec[j];
                    }
                }
                target_state_action_values[idx] = item.reward + max_next_state_value;
            }
        }
        paddle::framework::TensorFromVector(target_state_action_values, *dev_ctx_, &next_state_action_tensor);

        auto loss = paddle::tape::mean(paddle::tape::cross_entropy(action_values, next_state_action_values));
        paddle::tape::BackwardAndUpdate(loss, &adam_);
    }

    std::vector<float> inference(std::vector<float> state) {
        auto input = paddle::tape::VariableHandle(new paddle::tape::Variable("infer_state"));
        auto tensor = input->GetMutable<paddle::framework::Tensor>();
        fill_state_tensor(*dev_ctx_, state, tensor, 1);

        auto forward = [&](paddle::tape::VariableHandle input) -> paddle::tape::VariableHandle {
            return linear2_(linear1_(input));
        };

        forward(input);
    }

private:
    void fill_state_tensor(const paddle::platform::DeviceContext& ctx, std::vector<float> states, paddle::framework::Tensor* tensor, int batch_size) {
        paddle::framework::TensorFromVector(states, ctx, tensor);
        tensor->Resize(paddle::framework::make_ddim({batch_size, state_count_}));
    }
    
    paddle::tape::Linear linear1_;
    paddle::tape::Linear linear2_;
    paddle::tape::Adam adam_;

    int state_count_;
    int action_count_;
    paddle::platform::DeviceContext* dev_ctx_;
};

static
void run_single_environment(
        const std::shared_ptr<Gym::Client>& client,
        const std::string& env_id,
        int episodes_to_run)
{
        std::shared_ptr<Gym::Environment> env = client->make(env_id);
        std::shared_ptr<Gym::Space> action_space = env->action_space();
        std::shared_ptr<Gym::Space> observation_space = env->observation_space();
        ReplayBuffer buf(100);

        for (int e=0; e<episodes_to_run; ++e) {
                printf("%s episode %i...\n", env_id.c_str(), e);
                Gym::State s;
                env->reset(&s);
                float total_reward = 0;
                int total_steps = 0;
                while (1) {
                    Gym::State next_state;
                    std::vector<float> action = action_space->sample();
                    env->step(action, false, &next_state);
                    std::copy(next_state.observation.begin(),next_state.observation.end(),std::ostream_iterator<float>(std::cout, " " ));
                    std::cout<<"\n";
                    
                    // auto ss = observation_space->sample();
                    // std::copy(ss.begin(),ss.end(),std::ostream_iterator<int>(std::cout, " " ));
                    // std::cout<<"\n";
                    
                    std::cout<< "reward: " << next_state.reward << "\n";
                    struct Item item = {s.observation, action, next_state.reward, next_state.observation};
                    buf.add(item);
                    total_reward += next_state.reward;
                    total_steps += 1;
                    if (next_state.done) break;
                }
                printf("%s episode %i finished in %i steps with reward %0.2f\n",
                        env_id.c_str(), e, total_steps, total_reward);
        }
}

int main(int argc, char** argv)
{
        try {
                std::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
                run_single_environment(client, "CartPole-v0", 3);

        } catch (const std::exception& e) {
                fprintf(stderr, "ERROR: %s\n", e.what());
                return 1;
        }

        return 0;
}
