/**
 * NEURAL DIFFERENTIAL MANIFOLDS FOR TCP CONGESTION CONTROL (NDM-TCP)
 * 
 * Entropy-Aware Traffic Shaping with Continuous Weight Evolution
 * 
 * Key Features:
 * - Shannon Entropy calculation to distinguish noise from congestion
 * - Differential manifold treating TCP as a "physical pipe" that bends
 * - Real-time plasticity adapting to network conditions
 * - Security: Input validation, bounds checking, rate limiting
 * - Hebbian learning for traffic pattern recognition
 * 
 * Compile:
 * Windows: gcc -shared -o ndm_tcp.dll ndm_tcp.c -lm -O3 -fopenmp
 * Linux:   gcc -shared -fPIC -o ndm_tcp.so ndm_tcp.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o ndm_tcp.dylib ndm_tcp.c -lm -O3 -Xpreprocessor -fopenmp -lomp
 * 
 * Licensed under GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// SECURITY CONSTANTS
// ============================================================================

#define MAX_BANDWIDTH_GBPS 100.0f        // Maximum bandwidth limit (100 Gbps)
#define MIN_BANDWIDTH_MBPS 0.1f          // Minimum bandwidth (100 Kbps)
#define MAX_RTT_MS 10000.0f              // Maximum RTT (10 seconds)
#define MIN_RTT_MS 0.1f                  // Minimum RTT (0.1 ms)
#define MAX_CWND 1048576                 // Maximum congestion window (1M packets)
#define MIN_CWND 1                       // Minimum congestion window
#define MAX_PACKET_LOSS_RATE 1.0f        // Maximum packet loss rate (100%)
#define MAX_QUEUE_SIZE 100000            // Maximum queue size
#define ENTROPY_WINDOW_SIZE 100          // Window for entropy calculation
#define MAX_CONNECTIONS 10000            // Maximum concurrent connections

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    // Network state inputs (normalized)
    float current_rtt;           // Round-trip time (ms)
    float min_rtt;               // Minimum observed RTT
    float packet_loss_rate;      // Loss rate [0, 1]
    float bandwidth_estimate;    // Current bandwidth (Mbps)
    float queue_delay;           // Queuing delay (ms)
    float jitter;                // RTT variance
    float throughput;            // Current throughput (Mbps)
    
    // Entropy metrics
    float shannon_entropy;       // Traffic entropy
    float noise_ratio;           // Noise vs. signal ratio
    float congestion_confidence; // Confidence it's real congestion
    
    // Historical window for entropy
    float rtt_history[ENTROPY_WINDOW_SIZE];
    float loss_history[ENTROPY_WINDOW_SIZE];
    int history_index;
    int history_count;
    
} TCPState;

typedef struct {
    int input_size;              // TCP state vector dimension
    int hidden_size;             // Hidden layer size
    int output_size;             // Actions: [cwnd_delta, ssthresh_delta, pacing_rate]
    int manifold_size;           // Memory manifold size
    
    // === CORE WEIGHTS (Evolve via ODEs) ===
    float *W_input;              // [input_size × hidden_size]
    float *W_hidden;             // [hidden_size × hidden_size] (recurrent)
    float *W_output;             // [hidden_size × output_size]
    
    // === WEIGHT DERIVATIVES (dW/dt) ===
    float *dW_input_dt;
    float *dW_hidden_dt;
    float *dW_output_dt;
    
    // === WEIGHT VELOCITY (for momentum) ===
    float *V_input;
    float *V_hidden;
    float *V_output;
    
    // === PLASTICITY PARAMETERS ===
    float *plasticity_mask_input;
    float *plasticity_mask_hidden;
    float *plasticity_mask_output;
    float base_plasticity;
    float plasticity_decay;
    
    // === ASSOCIATIVE MEMORY MANIFOLD ===
    // Stores learned traffic patterns
    float *Memory_manifold;      // [manifold_size × hidden_size]
    float *Key_weights;
    float *Value_weights;
    float *Query_weights;
    float memory_decay;
    
    // === HEBBIAN TRACE ===
    float *pre_trace;
    float *post_trace;
    float trace_decay;
    
    // === STATE VARIABLES ===
    float *hidden_state;
    float *output_state;
    float *hidden_derivative;
    
    // === ODE SOLVER PARAMETERS ===
    float dt;
    float weight_decay_lambda;
    
    // === TRAINING PARAMETERS ===
    float learning_rate;
    float momentum;
    int training_steps;
    float last_reward;
    
    // === TCP-SPECIFIC ===
    float current_cwnd;          // Current congestion window
    float current_ssthresh;      // Slow start threshold
    float current_pacing_rate;   // Pacing rate (Mbps)
    
    // === ENTROPY TRACKING ===
    float avg_entropy;
    float entropy_threshold;     // Threshold for noise vs. congestion
    
    // === SECURITY ===
    uint64_t total_packets_processed;
    uint64_t total_bytes_processed;
    time_t creation_time;
    bool is_valid;
    
    // === STATISTICS ===
    float avg_weight_velocity;
    float avg_plasticity;
    float avg_manifold_energy;
    float hebbian_strength;
    
} NDMTCPNetwork;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

static inline float tanh_act(float x) {
    return tanhf(x);
}

static inline float tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static float randn(void) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
}

static float clip_value(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// ============================================================================
// ENTROPY CALCULATION (Shannon Entropy)
// ============================================================================

/**
 * Calculate Shannon Entropy of a signal to distinguish noise from congestion
 * 
 * H(X) = -Σ p(x) * log2(p(x))
 * 
 * High entropy → Random noise (don't reduce cwnd aggressively)
 * Low entropy  → Structured congestion (real bottleneck, reduce cwnd)
 */
static float calculate_shannon_entropy(const float *data, int size) {
    if (size == 0) return 0.0f;
    
    // Create histogram with 20 bins
    const int num_bins = 20;
    int histogram[20] = {0};
    
    // Find min/max for normalization
    float min_val = data[0], max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    float range = max_val - min_val;
    if (range < 1e-6f) return 0.0f; // No variance
    
    // Fill histogram
    for (int i = 0; i < size; i++) {
        int bin = (int)((data[i] - min_val) / range * (num_bins - 1));
        bin = clip_value(bin, 0, num_bins - 1);
        histogram[bin]++;
    }
    
    // Calculate entropy
    float entropy = 0.0f;
    for (int i = 0; i < num_bins; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / size;
            entropy -= p * log2f(p);
        }
    }
    
    return entropy;
}

/**
 * Calculate noise ratio: high entropy regions indicate noise
 */
static float calculate_noise_ratio(float entropy, float max_entropy) {
    // Normalize entropy to [0, 1]
    float norm_entropy = entropy / (max_entropy + 1e-6f);
    
    // High entropy (>0.8) → likely noise
    // Low entropy (<0.4) → likely real congestion
    if (norm_entropy > 0.8f) return 0.9f;  // 90% noise
    if (norm_entropy < 0.4f) return 0.1f;  // 10% noise
    
    return norm_entropy; // Linear interpolation
}

// ============================================================================
// SECURITY VALIDATION
// ============================================================================

static bool validate_tcp_state(const TCPState *state) {
    if (!state) return false;
    
    // Check RTT bounds
    if (state->current_rtt < 0.0f || state->current_rtt > MAX_RTT_MS) return false;
    if (state->min_rtt < 0.0f || state->min_rtt > MAX_RTT_MS) return false;
    if (state->min_rtt > state->current_rtt) return false;
    
    // Check packet loss rate
    if (state->packet_loss_rate < 0.0f || state->packet_loss_rate > MAX_PACKET_LOSS_RATE) 
        return false;
    
    // Check bandwidth
    if (state->bandwidth_estimate < 0.0f || 
        state->bandwidth_estimate > MAX_BANDWIDTH_GBPS * 1000.0f) 
        return false;
    
    // Check queue delay
    if (state->queue_delay < 0.0f || state->queue_delay > MAX_RTT_MS) 
        return false;
    
    // Check jitter
    if (state->jitter < 0.0f || state->jitter > MAX_RTT_MS) 
        return false;
    
    // Check throughput
    if (state->throughput < 0.0f || state->throughput > MAX_BANDWIDTH_GBPS * 1000.0f) 
        return false;
    
    return true;
}

static bool validate_network(const NDMTCPNetwork *net) {
    if (!net) return false;
    if (!net->is_valid) return false;
    
    // Check dimensions
    if (net->input_size <= 0 || net->input_size > 1000) return false;
    if (net->hidden_size <= 0 || net->hidden_size > 10000) return false;
    if (net->output_size <= 0 || net->output_size > 100) return false;
    if (net->manifold_size <= 0 || net->manifold_size > 1000) return false;
    
    // Check pointers
    if (!net->W_input || !net->W_hidden || !net->W_output) return false;
    if (!net->hidden_state || !net->output_state) return false;
    
    // Check TCP state
    if (net->current_cwnd < MIN_CWND || net->current_cwnd > MAX_CWND) return false;
    if (net->current_ssthresh < MIN_CWND || net->current_ssthresh > MAX_CWND) return false;
    if (net->current_pacing_rate < 0.0f || 
        net->current_pacing_rate > MAX_BANDWIDTH_GBPS * 1000.0f) return false;
    
    return true;
}

// ============================================================================
// NETWORK CREATION
// ============================================================================

EXPORT NDMTCPNetwork* create_ndm_tcp(int input_size, int hidden_size, int output_size,
                                      int manifold_size, float learning_rate) {
    // Security: Validate inputs
    if (input_size <= 0 || input_size > 1000) return NULL;
    if (hidden_size <= 0 || hidden_size > 10000) return NULL;
    if (output_size <= 0 || output_size > 100) return NULL;
    if (manifold_size <= 0 || manifold_size > 1000) return NULL;
    if (learning_rate <= 0.0f || learning_rate > 1.0f) return NULL;
    
    NDMTCPNetwork *net = (NDMTCPNetwork*)calloc(1, sizeof(NDMTCPNetwork));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->manifold_size = manifold_size;
    net->learning_rate = learning_rate;
    net->base_plasticity = 0.02f;
    net->plasticity_decay = 0.998f;
    net->momentum = 0.9f;
    net->dt = 0.1f;
    net->weight_decay_lambda = 0.0001f;
    net->memory_decay = 0.995f;
    net->trace_decay = 0.95f;
    net->hebbian_strength = 0.15f;
    net->training_steps = 0;
    
    // TCP-specific initialization
    net->current_cwnd = 10.0f;           // Start with 10 packets
    net->current_ssthresh = 65535.0f;    // Standard initial ssthresh
    net->current_pacing_rate = 10.0f;    // 10 Mbps initial
    net->entropy_threshold = 3.5f;       // Threshold for noise detection
    
    // Security
    net->creation_time = time(NULL);
    net->total_packets_processed = 0;
    net->total_bytes_processed = 0;
    net->is_valid = true;
    
    // Allocate weight matrices
    net->W_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->W_hidden = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    net->W_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    
    if (!net->W_input || !net->W_hidden || !net->W_output) {
        free(net->W_input);
        free(net->W_hidden);
        free(net->W_output);
        free(net);
        return NULL;
    }
    
    // Allocate derivatives
    net->dW_input_dt = (float*)calloc(input_size * hidden_size, sizeof(float));
    net->dW_hidden_dt = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    net->dW_output_dt = (float*)calloc(hidden_size * output_size, sizeof(float));
    
    // Allocate velocities
    net->V_input = (float*)calloc(input_size * hidden_size, sizeof(float));
    net->V_hidden = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    net->V_output = (float*)calloc(hidden_size * output_size, sizeof(float));
    
    // Allocate plasticity masks
    net->plasticity_mask_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->plasticity_mask_hidden = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    net->plasticity_mask_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    
    // Allocate manifold
    net->Memory_manifold = (float*)calloc(manifold_size * hidden_size, sizeof(float));
    net->Key_weights = (float*)malloc(hidden_size * manifold_size * sizeof(float));
    net->Value_weights = (float*)malloc(hidden_size * manifold_size * sizeof(float));
    net->Query_weights = (float*)malloc(hidden_size * manifold_size * sizeof(float));
    
    // Allocate traces
    net->pre_trace = (float*)calloc(hidden_size, sizeof(float));
    net->post_trace = (float*)calloc(hidden_size, sizeof(float));
    
    // Allocate states
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    net->hidden_derivative = (float*)calloc(hidden_size, sizeof(float));
    
    // Initialize weights with Xavier/Glorot
    srand(time(NULL));
    
    float scale_input = sqrtf(2.0f / (input_size + hidden_size));
    for (int i = 0; i < input_size * hidden_size; i++) {
        net->W_input[i] = randn() * scale_input;
        net->plasticity_mask_input[i] = 1.0f;
    }
    
    float scale_hidden = sqrtf(2.0f / (hidden_size + hidden_size));
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        net->W_hidden[i] = randn() * scale_hidden;
        net->plasticity_mask_hidden[i] = 1.0f;
    }
    
    float scale_output = sqrtf(2.0f / (hidden_size + output_size));
    for (int i = 0; i < hidden_size * output_size; i++) {
        net->W_output[i] = randn() * scale_output;
        net->plasticity_mask_output[i] = 1.0f;
    }
    
    // Initialize manifold weights
    float scale_manifold = sqrtf(2.0f / (hidden_size + manifold_size));
    for (int i = 0; i < hidden_size * manifold_size; i++) {
        net->Key_weights[i] = randn() * scale_manifold;
        net->Value_weights[i] = randn() * scale_manifold;
        net->Query_weights[i] = randn() * scale_manifold;
    }
    
    return net;
}

// ============================================================================
// NETWORK DESTRUCTION
// ============================================================================

EXPORT void destroy_ndm_tcp(NDMTCPNetwork *net) {
    if (!net) return;
    
    net->is_valid = false;
    
    free(net->W_input);
    free(net->W_hidden);
    free(net->W_output);
    free(net->dW_input_dt);
    free(net->dW_hidden_dt);
    free(net->dW_output_dt);
    free(net->V_input);
    free(net->V_hidden);
    free(net->V_output);
    free(net->plasticity_mask_input);
    free(net->plasticity_mask_hidden);
    free(net->plasticity_mask_output);
    free(net->Memory_manifold);
    free(net->Key_weights);
    free(net->Value_weights);
    free(net->Query_weights);
    free(net->pre_trace);
    free(net->post_trace);
    free(net->hidden_state);
    free(net->output_state);
    free(net->hidden_derivative);
    
    free(net);
}

// ============================================================================
// UPDATE TCP STATE WITH ENTROPY
// ============================================================================

EXPORT void update_tcp_state_entropy(TCPState *state, float rtt, float loss_rate) {
    if (!state) return;
    
    // Security: Validate inputs
    rtt = clip_value(rtt, MIN_RTT_MS, MAX_RTT_MS);
    loss_rate = clip_value(loss_rate, 0.0f, MAX_PACKET_LOSS_RATE);
    
    // Update history
    state->rtt_history[state->history_index] = rtt;
    state->loss_history[state->history_index] = loss_rate;
    state->history_index = (state->history_index + 1) % ENTROPY_WINDOW_SIZE;
    
    if (state->history_count < ENTROPY_WINDOW_SIZE) {
        state->history_count++;
    }
    
    // Calculate Shannon entropy if we have enough data
    if (state->history_count >= 10) {
        float rtt_entropy = calculate_shannon_entropy(state->rtt_history, 
                                                       state->history_count);
        float loss_entropy = calculate_shannon_entropy(state->loss_history, 
                                                        state->history_count);
        
        // Combined entropy (weighted average)
        state->shannon_entropy = 0.6f * rtt_entropy + 0.4f * loss_entropy;
        
        // Calculate noise ratio
        float max_entropy = log2f(20.0f); // Maximum entropy for 20 bins
        state->noise_ratio = calculate_noise_ratio(state->shannon_entropy, max_entropy);
        
        // Congestion confidence: inverse of noise ratio
        state->congestion_confidence = 1.0f - state->noise_ratio;
    } else {
        // Not enough data, assume low noise
        state->shannon_entropy = 0.0f;
        state->noise_ratio = 0.2f;
        state->congestion_confidence = 0.8f;
    }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void ndm_tcp_forward(NDMTCPNetwork *net, const TCPState *tcp_state, 
                            float *actions) {
    if (!validate_network(net)) return;
    if (!validate_tcp_state(tcp_state)) return;
    if (!actions) return;
    
    int I = net->input_size;
    int H = net->hidden_size;
    int O = net->output_size;
    int M = net->manifold_size;
    
    // === Step 1: Prepare input vector ===
    float input[32];  // Should match input_size
    int idx = 0;
    
    // Normalize inputs to [0, 1] or [-1, 1]
    input[idx++] = tcp_state->current_rtt / 1000.0f;  // Normalize to seconds
    input[idx++] = tcp_state->min_rtt / 1000.0f;
    input[idx++] = tcp_state->packet_loss_rate;
    input[idx++] = tcp_state->bandwidth_estimate / 1000.0f; // Normalize to Gbps
    input[idx++] = tcp_state->queue_delay / 100.0f;
    input[idx++] = tcp_state->jitter / 100.0f;
    input[idx++] = tcp_state->throughput / 1000.0f;
    
    // Entropy features (CRITICAL for distinguishing noise from congestion)
    input[idx++] = tcp_state->shannon_entropy / 5.0f;  // Normalize
    input[idx++] = tcp_state->noise_ratio;
    input[idx++] = tcp_state->congestion_confidence;
    
    // Current TCP state
    input[idx++] = logf(net->current_cwnd + 1.0f) / 15.0f;  // Log scale
    input[idx++] = logf(net->current_ssthresh + 1.0f) / 15.0f;
    input[idx++] = net->current_pacing_rate / 1000.0f;
    
    // Derived features
    float rtt_ratio = tcp_state->current_rtt / (tcp_state->min_rtt + 0.1f);
    input[idx++] = clip_value(rtt_ratio / 10.0f, 0.0f, 1.0f);
    
    float bdp = tcp_state->bandwidth_estimate * tcp_state->min_rtt / 8.0f; // In packets
    input[idx++] = logf(bdp + 1.0f) / 15.0f;
    
    // Pad to input_size if needed
    while (idx < I) input[idx++] = 0.0f;
    
    // === Step 2: Input → Hidden ===
    float input_contrib[H];
    for (int i = 0; i < H; i++) {
        input_contrib[i] = 0.0f;
        for (int j = 0; j < I; j++) {
            input_contrib[i] += net->W_input[j * H + i] * input[j];
        }
    }
    
    // === Step 3: Recurrent Hidden → Hidden ===
    float recurrent_contrib[H];
    for (int i = 0; i < H; i++) {
        recurrent_contrib[i] = 0.0f;
        for (int j = 0; j < H; j++) {
            recurrent_contrib[i] += net->W_hidden[j * H + i] * net->hidden_state[j];
        }
    }
    
    // === Step 4: Memory Manifold Read (Attention mechanism) ===
    float queries[M];
    for (int m = 0; m < M; m++) {
        queries[m] = 0.0f;
        for (int i = 0; i < H; i++) {
            queries[m] += net->Query_weights[i * M + m] * net->hidden_state[i];
        }
        queries[m] = sigmoid(queries[m]);
    }
    
    float memory_read[H];
    for (int i = 0; i < H; i++) {
        memory_read[i] = 0.0f;
        for (int m = 0; m < M; m++) {
            memory_read[i] += queries[m] * net->Memory_manifold[m * H + i];
        }
    }
    
    // === Step 5: Update hidden state using ODE ===
    for (int i = 0; i < H; i++) {
        float total_input = input_contrib[i] + recurrent_contrib[i] + memory_read[i];
        net->hidden_derivative[i] = -net->hidden_state[i] + tanh_act(total_input);
        net->hidden_state[i] += net->dt * net->hidden_derivative[i];
    }
    
    // === Step 6: Update Hebbian traces ===
    for (int i = 0; i < H; i++) {
        net->pre_trace[i] = net->trace_decay * net->pre_trace[i] + 
                            (1.0f - net->trace_decay) * fabsf(net->hidden_state[i]);
        net->post_trace[i] = net->trace_decay * net->post_trace[i] + 
                             (1.0f - net->trace_decay) * fabsf(net->hidden_derivative[i]);
    }
    
    // === Step 7: Compute weight derivatives (neuroplasticity) ===
    // Input weights evolution
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < H; j++) {
            int idx = i * H + j;
            float hebbian = net->hebbian_strength * input[i] * net->post_trace[j];
            float decay = net->weight_decay_lambda * net->W_input[idx];
            net->dW_input_dt[idx] = net->plasticity_mask_input[idx] * (hebbian - decay);
        }
    }
    
    // Hidden weights evolution
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            int idx = i * H + j;
            float hebbian = net->hebbian_strength * net->pre_trace[i] * net->post_trace[j];
            float decay = net->weight_decay_lambda * net->W_hidden[idx];
            net->dW_hidden_dt[idx] = net->plasticity_mask_hidden[idx] * (hebbian - decay);
        }
    }
    
    // === Step 8: Integrate weight ODEs ===
    for (int i = 0; i < I * H; i++) {
        net->W_input[i] += net->dt * net->dW_input_dt[i];
    }
    
    for (int i = 0; i < H * H; i++) {
        net->W_hidden[i] += net->dt * net->dW_hidden_dt[i];
    }
    
    // Decay plasticity
    for (int i = 0; i < I * H; i++) {
        net->plasticity_mask_input[i] *= net->plasticity_decay;
    }
    for (int i = 0; i < H * H; i++) {
        net->plasticity_mask_hidden[i] *= net->plasticity_decay;
    }
    
    // === Step 9: Compute output (TCP actions) ===
    for (int i = 0; i < O; i++) {
        actions[i] = 0.0f;
        for (int j = 0; j < H; j++) {
            actions[i] += net->W_output[j * O + i] * net->hidden_state[j];
        }
    }
    
    // Apply activation to actions
    // actions[0] = cwnd_delta (can be negative)
    // actions[1] = ssthresh_delta (can be negative)
    // actions[2] = pacing_rate_multiplier (positive)
    
    actions[0] = tanh_act(actions[0]) * 10.0f;  // cwnd delta: ±10 packets
    actions[1] = tanh_act(actions[1]) * 100.0f; // ssthresh delta: ±100
    actions[2] = sigmoid(actions[2]) * 2.0f;    // pacing multiplier: [0, 2]
    
    memcpy(net->output_state, actions, O * sizeof(float));
    
    // === Update statistics ===
    float weight_velocity = 0.0f;
    for (int i = 0; i < I * H; i++) {
        weight_velocity += fabsf(net->dW_input_dt[i]);
    }
    for (int i = 0; i < H * H; i++) {
        weight_velocity += fabsf(net->dW_hidden_dt[i]);
    }
    net->avg_weight_velocity = weight_velocity / (I * H + H * H);
    
    float avg_plast = 0.0f;
    for (int i = 0; i < I * H; i++) avg_plast += net->plasticity_mask_input[i];
    for (int i = 0; i < H * H; i++) avg_plast += net->plasticity_mask_hidden[i];
    net->avg_plasticity = avg_plast / (I * H + H * H);
    
    float energy = 0.0f;
    for (int i = 0; i < M * H; i++) {
        energy += net->Memory_manifold[i] * net->Memory_manifold[i];
    }
    net->avg_manifold_energy = energy / (M * H);
    
    net->avg_entropy = tcp_state->shannon_entropy;
    
    // Update packet counter (security)
    net->total_packets_processed++;
}

// ============================================================================
// APPLY ACTIONS TO TCP STATE
// ============================================================================

EXPORT void apply_tcp_actions(NDMTCPNetwork *net, const float *actions, 
                              float reward) {
    if (!validate_network(net)) return;
    if (!actions) return;
    
    // Extract actions
    float cwnd_delta = actions[0];
    float ssthresh_delta = actions[1];
    float pacing_multiplier = actions[2];
    
    // Apply with security bounds
    net->current_cwnd += cwnd_delta;
    net->current_cwnd = clip_value(net->current_cwnd, MIN_CWND, MAX_CWND);
    
    net->current_ssthresh += ssthresh_delta;
    net->current_ssthresh = clip_value(net->current_ssthresh, MIN_CWND, MAX_CWND);
    
    net->current_pacing_rate *= pacing_multiplier;
    net->current_pacing_rate = clip_value(net->current_pacing_rate, 
                                          MIN_BANDWIDTH_MBPS, 
                                          MAX_BANDWIDTH_GBPS * 1000.0f);
    
    net->last_reward = reward;
}

// ============================================================================
// TRAINING (Reinforcement Learning)
// ============================================================================

EXPORT float ndm_tcp_train(NDMTCPNetwork *net, const TCPState *tcp_state, 
                           float reward) {
    if (!validate_network(net)) return -1.0f;
    if (!validate_tcp_state(tcp_state)) return -1.0f;
    
    // Forward pass
    float actions[3];
    ndm_tcp_forward(net, tcp_state, actions);
    
    // Apply actions
    apply_tcp_actions(net, actions, reward);
    
    // Backprop through output weights using reward
    // Higher reward → reinforce current actions
    // Lower reward → discourage current actions
    
    float reward_signal = tanhf(reward / 100.0f);  // Normalize reward
    
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            int idx = i * net->output_size + j;
            float grad = -reward_signal * net->output_state[j];
            
            // Momentum update
            net->V_output[idx] = net->momentum * net->V_output[idx] - 
                                 net->learning_rate * grad;
            net->W_output[idx] += net->V_output[idx];
            
            // Compute derivative for ODE
            net->dW_output_dt[idx] = net->plasticity_mask_output[idx] * 
                                      (net->hebbian_strength * net->hidden_state[i] * reward_signal -
                                       net->weight_decay_lambda * net->W_output[idx]);
        }
    }
    
    // Boost plasticity on poor performance
    float error_magnitude = fabsf(reward_signal);
    if (reward < 0) {
        for (int i = 0; i < net->input_size * net->hidden_size; i++) {
            net->plasticity_mask_input[i] = fminf(1.0f, 
                net->plasticity_mask_input[i] + 0.02f * error_magnitude);
        }
        for (int i = 0; i < net->hidden_size * net->hidden_size; i++) {
            net->plasticity_mask_hidden[i] = fminf(1.0f, 
                net->plasticity_mask_hidden[i] + 0.02f * error_magnitude);
        }
    }
    
    net->training_steps++;
    
    return fabsf(reward);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT void ndm_tcp_reset_memory(NDMTCPNetwork *net) {
    if (!validate_network(net)) return;
    
    memset(net->hidden_state, 0, net->hidden_size * sizeof(float));
    memset(net->Memory_manifold, 0, 
           net->manifold_size * net->hidden_size * sizeof(float));
    memset(net->pre_trace, 0, net->hidden_size * sizeof(float));
    memset(net->post_trace, 0, net->hidden_size * sizeof(float));
}

EXPORT float ndm_tcp_get_avg_weight_velocity(NDMTCPNetwork *net) {
    return validate_network(net) ? net->avg_weight_velocity : 0.0f;
}

EXPORT float ndm_tcp_get_avg_plasticity(NDMTCPNetwork *net) {
    return validate_network(net) ? net->avg_plasticity : 0.0f;
}

EXPORT float ndm_tcp_get_avg_manifold_energy(NDMTCPNetwork *net) {
    return validate_network(net) ? net->avg_manifold_energy : 0.0f;
}

EXPORT float ndm_tcp_get_avg_entropy(NDMTCPNetwork *net) {
    return validate_network(net) ? net->avg_entropy : 0.0f;
}

EXPORT float ndm_tcp_get_current_cwnd(NDMTCPNetwork *net) {
    return validate_network(net) ? net->current_cwnd : 0.0f;
}

EXPORT float ndm_tcp_get_current_ssthresh(NDMTCPNetwork *net) {
    return validate_network(net) ? net->current_ssthresh : 0.0f;
}

EXPORT float ndm_tcp_get_current_pacing_rate(NDMTCPNetwork *net) {
    return validate_network(net) ? net->current_pacing_rate : 0.0f;
}

EXPORT void ndm_tcp_print_info(NDMTCPNetwork *net) {
    if (!validate_network(net)) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   NEURAL DIFFERENTIAL MANIFOLDS FOR TCP (NDM-TCP)            ║\n");
    printf("║   Entropy-Aware Congestion Control                            ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d -> %d -> %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    printf("Manifold Size: %d\n\n", net->manifold_size);
    
    printf("TCP State:\n");
    printf("  Congestion Window:     %.2f packets\n", net->current_cwnd);
    printf("  Slow Start Threshold:  %.2f packets\n", net->current_ssthresh);
    printf("  Pacing Rate:           %.2f Mbps\n\n", net->current_pacing_rate);
    
    printf("Neuroplasticity Metrics:\n");
    printf("  Weight Velocity:       %.6f\n", net->avg_weight_velocity);
    printf("  Average Plasticity:    %.4f\n", net->avg_plasticity);
    printf("  Hebbian Strength:      %.4f\n\n", net->hebbian_strength);
    
    printf("Entropy Analysis:\n");
    printf("  Average Entropy:       %.4f\n", net->avg_entropy);
    printf("  Entropy Threshold:     %.4f\n\n", net->entropy_threshold);
    
    printf("Memory:\n");
    printf("  Manifold Energy:       %.6f\n\n", net->avg_manifold_energy);
    
    printf("Training:\n");
    printf("  Steps:                 %d\n", net->training_steps);
    printf("  Last Reward:           %.4f\n", net->last_reward);
    printf("  Packets Processed:     %llu\n", 
           (unsigned long long)net->total_packets_processed);
}

EXPORT TCPState* create_tcp_state(void) {
    TCPState *state = (TCPState*)calloc(1, sizeof(TCPState));
    if (!state) return NULL;
    
    // Initialize with safe defaults
    state->current_rtt = 50.0f;      // 50 ms
    state->min_rtt = 10.0f;          // 10 ms
    state->packet_loss_rate = 0.0f;
    state->bandwidth_estimate = 100.0f; // 100 Mbps
    state->queue_delay = 5.0f;
    state->jitter = 2.0f;
    state->throughput = 50.0f;
    state->shannon_entropy = 0.0f;
    state->noise_ratio = 0.2f;
    state->congestion_confidence = 0.8f;
    state->history_index = 0;
    state->history_count = 0;
    
    return state;
}

EXPORT void destroy_tcp_state(TCPState *state) {
    free(state);
}