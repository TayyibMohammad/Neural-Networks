#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric> // For std::accumulate

// Using a global random device and engine for simplicity.
// For more robust applications, pass these around or make them members of a class.
std::random_device rd;
std::mt19937 gen(rd()); // Seed with a random device
std::uniform_real_distribution<> distrib(-1.0, 1.0); // Distribution for weights and biases

class LayerDense {
public:
    int n_inputs;
    int n_outputs;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    LayerDense(int n_inputs_val, int n_outputs_val) :
        n_inputs(n_inputs_val),
        n_outputs(n_outputs_val) {

        // Initialize weights
        weights.resize(n_inputs);
        for (int i = 0; i < n_inputs; ++i) {
            weights[i].resize(n_outputs);
            for (int j = 0; j < n_outputs; ++j) {
                weights[i][j] = distrib(gen); // Random weight between -1 and 1
            }
        }

        // Initialize biases
        biases.resize(n_outputs, 0.0); // All biases initialized to 0
    }

    // Function to get current parameters (weights and biases)
    std::pair<std::vector<std::vector<double>>, std::vector<double>> getParams() const {
        return {weights, biases};
    }

    // Forward pass calculation
    std::vector<double> forward(const std::vector<double>& inputs,
                                const std::vector<std::vector<double>>& wts,
                                const std::vector<double>& bia) const {
        std::vector<double> outputs(n_outputs, 0.0); // Initialize outputs with zeros

        for (int i = 0; i < n_outputs; ++i) { // Iterate over outputs (columns of weights)
            double output_val = 0.0;
            for (int j = 0; j < n_inputs; ++j) { // Iterate over inputs (rows of weights)
                output_val += wts[j][i] * inputs[j];
            }
            outputs[i] = output_val + bia[i];
        }
        return outputs;
    }

    // Reset parameters by adding small random perturbations
    std::pair<std::vector<std::vector<double>>, std::vector<double>> resetParams() const {
        std::vector<std::vector<double>> new_weights = weights; // Copy current weights
        std::vector<double> new_biases = biases; // Copy current biases

        // Perturb weights
        for (int i = 0; i < n_inputs; ++i) {
            for (int j = 0; j < n_outputs; ++j) {
                new_weights[i][j] += distrib(gen); // Add random perturbation
            }
        }

        // Perturb biases
        for (int i = 0; i < n_outputs; ++i) {
            new_biases[i] += distrib(gen); // Add random perturbation
        }

        return {new_weights, new_biases};
    }
};

class ActivationRelu {
public:
    std::vector<double> forward(const std::vector<double>& inputs) const {
        std::vector<double> outputs;
        outputs.reserve(inputs.size()); // Pre-allocate memory for efficiency
        for (double input : inputs) {
            outputs.push_back(std::max(0.0, input));
        }
        return outputs;
    }
};

class SoftmaxActivation {
public:
    std::vector<double> forward(const std::vector<double>& inputs) const {
        std::vector<double> exp_inputs;
        exp_inputs.reserve(inputs.size());
        double total_exp = 0.0;

        // Calculate exponentials
        for (double input : inputs) {
            double val = std::exp(input);
            exp_inputs.push_back(val);
            total_exp += val;
        }

        // Normalize
        std::vector<double> outputs;
        outputs.reserve(inputs.size());
        for (double val : exp_inputs) {
            outputs.push_back(val / total_exp);
        }
        return outputs;
    }
};

class SigmoidActivation {
public:
    std::vector<double> forward(const std::vector<double>& inputs) const {
        std::vector<double> outputs;
        outputs.reserve(inputs.size());
        for (double input : inputs) {
            outputs.push_back(1.0 / (1.0 + std::exp(-input)));
        }
        return outputs;
    }
};

// Function to calculate the forward pass for the entire network
std::vector<std::vector<double>> calc(
    const std::vector<std::vector<double>>& X_data,
    const LayerDense& layer1,
    const std::vector<std::vector<double>>& layer1_wts,
    const std::vector<double>& layer1_bia,
    const LayerDense& layer2,
    const std::vector<std::vector<double>>& layer2_wts,
    const std::vector<double>& layer2_bia,
    const LayerDense& layer3,
    const std::vector<std::vector<double>>& layer3_wts,
    const std::vector<double>& layer3_bia
) {
    ActivationRelu relu_activation;
    SoftmaxActivation softmax_activation;
    std::vector<std::vector<double>> y_pred;

    for (const auto& input_sample : X_data) {
        std::vector<double> output1 = layer1.forward(input_sample, layer1_wts, layer1_bia);
        std::vector<double> activated_output1 = relu_activation.forward(output1);

        std::vector<double> output2 = layer2.forward(activated_output1, layer2_wts, layer2_bia);
        std::vector<double> activated_output2 = relu_activation.forward(output2);

        std::vector<double> output3 = layer3.forward(activated_output2, layer3_wts, layer3_bia);
        std::vector<double> output4 = softmax_activation.forward(output3);
        y_pred.push_back(output4);
    }
    return y_pred;
}

// Function to calculate cross-entropy error
double CalculateError(const std::vector<std::vector<int>>& y_true,
                      const std::vector<std::vector<double>>& y_pred) {
    double error = 0.0;
    int num_classes = y_true[0].size(); // Assuming all y_true have same number of classes

    for (size_t i = 0; i < y_pred.size(); ++i) {
        for (int j = 0; j < num_classes; ++j) {
            // Avoid log(0) issues by clamping y_pred[i][j]
            double pred_val = y_pred[i][j];
            if (pred_val < 1e-9) { // Small epsilon to prevent log(0)
                pred_val = 1e-9;
            }
            error += (y_true[i][j] * std::log(pred_val));
        }
    }
    return -1.0 * error / y_pred.size(); // Average error
}


int main() {
    // Instantiate layers
    LayerDense layer1(2, 3);
    auto params1 = layer1.getParams();
    std::vector<std::vector<double>> layer1_wts = params1.first;
    std::vector<double> layer1_bia = params1.second;

    LayerDense layer2(3, 3);
    auto params2 = layer2.getParams();
    std::vector<std::vector<double>> layer2_wts = params2.first;
    std::vector<double> layer2_bia = params2.second;

    LayerDense layer3(3, 4);
    auto params3 = layer3.getParams();
    std::vector<std::vector<double>> layer3_wts = params3.first;
    std::vector<double> layer3_bia = params3.second;

    // ***************Data_Generation****************
    std::vector<std::vector<double>> X;
    for (int i = -10; i <= 10; ++i) {
        for (int k = -10; k <= 10; ++k) {
            if (i != 0 && k != 0) {
                std::vector<double> coord;
                coord.push_back(static_cast<double>(i));
                coord.push_back(static_cast<double>(k));
                X.push_back(coord);
            }
        }
    }

    std::vector<std::vector<int>> y;
    for (const auto& coord : X) {
        if (coord[0] > 0 && coord[1] > 0) {
            y.push_back({1, 0, 0, 0});
        } else if (coord[0] < 0 && coord[1] > 0) {
            y.push_back({0, 1, 0, 0});
        } else if (coord[0] < 0 && coord[1] < 0) {
            y.push_back({0, 0, 1, 0});
        } else if (coord[0] > 0 && coord[1] < 0) {
            y.push_back({0, 0, 0, 1});
        }
    }
    // ***************Data_Generation_Ended**************

    // Test a single forward pass
    std::vector<double> test_input = {0, 0};
    std::vector<double> output_test1 = layer1.forward(test_input, layer1_wts, layer1_bia);
    std::vector<double> output_test2 = layer2.forward(ActivationRelu().forward(output_test1), layer2_wts, layer2_bia);
    std::vector<double> output_test3 = layer3.forward(ActivationRelu().forward(output_test2), layer3_wts, layer3_bia);
    std::vector<double> output_test4 = SoftmaxActivation().forward(output_test3);

    std::cout << "Initial test prediction for [0, 0]: [";
    for (size_t i = 0; i < output_test4.size(); ++i) {
        std::cout << output_test4[i] << (i == output_test4.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // Training loop
    int n_iteration = 30000;
    for (int i = 0; i < n_iteration; ++i) {
        std::vector<std::vector<double>> y_pred = calc(X, layer1, layer1_wts, layer1_bia,
                                                         layer2, layer2_wts, layer2_bia,
                                                         layer3, layer3_wts, layer3_bia);
        double minError = CalculateError(y, y_pred);

        // Try perturbing Layer 1
        auto new_params1 = layer1.resetParams();
        std::vector<std::vector<double>> new_layer1_wts = new_params1.first;
        std::vector<double> new_layer1_bia = new_params1.second;
        y_pred = calc(X, layer1, new_layer1_wts, new_layer1_bia, // Use new layer1 params
                      layer2, layer2_wts, layer2_bia,
                      layer3, layer3_wts, layer3_bia);
        double currError = CalculateError(y, y_pred);
        if (currError < minError) {
            minError = currError;
            layer1_wts = new_layer1_wts;
            layer1_bia = new_layer1_bia;
            std::cout << minError << " (Layer 1 updated) " << i << std::endl;
        }

        // Try perturbing Layer 2
        auto new_params2 = layer2.resetParams();
        std::vector<std::vector<double>> new_layer2_wts = new_params2.first;
        std::vector<double> new_layer2_bia = new_params2.second;
        y_pred = calc(X, layer1, layer1_wts, layer1_bia,
                      layer2, new_layer2_wts, new_layer2_bia, // Use new layer2 params
                      layer3, layer3_wts, layer3_bia);
        currError = CalculateError(y, y_pred);
        if (currError < minError) {
            minError = currError;
            layer2_wts = new_layer2_wts;
            layer2_bia = new_layer2_bia;
            std::cout << minError << " (Layer 2 updated) " << i << std::endl;
        }

        // Try perturbing Layer 3
        auto new_params3 = layer3.resetParams();
        std::vector<std::vector<double>> new_layer3_wts = new_params3.first;
        std::vector<double> new_layer3_bia = new_params3.second;
        y_pred = calc(X, layer1, layer1_wts, layer1_bia,
                      layer2, layer2_wts, layer2_bia,
                      layer3, new_layer3_wts, new_layer3_bia); // Use new layer3 params
        currError = CalculateError(y, y_pred);
        if (currError < minError) {
            minError = currError;
            layer3_wts = new_layer3_wts;
            layer3_bia = new_layer3_bia;
            std::cout << minError << " (Layer 3 updated) " << i << std::endl;
        }
    }

    // Final prediction after training
    output_test1 = layer1.forward(test_input, layer1_wts, layer1_bia);
    output_test2 = layer2.forward(ActivationRelu().forward(output_test1), layer2_wts, layer2_bia);
    output_test3 = layer3.forward(ActivationRelu().forward(output_test2), layer3_wts, layer3_bia);
    output_test4 = SoftmaxActivation().forward(output_test3);

    std::cout << "\nFinal test prediction for [60, -16.598]: [";
    for (size_t i = 0; i < output_test4.size(); ++i) {
        std::cout << output_test4[i] << (i == output_test4.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // Print final weights and biases (optional, can be very large)
    /*
    std::cout << "\nFinal Layer 1 Weights:" << std::endl;
    for (const auto& row : layer1_wts) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Final Layer 1 Biases: ";
    for (double val : layer1_bia) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    // ... similarly for layer 2 and 3
    */

    return 0;
}
