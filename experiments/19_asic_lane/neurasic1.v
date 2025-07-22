// File: neurasic1.v
// A minimal single neuron module with ReLU activation
// Inputs: weight, input_value, bias
// Output: activated_output = max(0, input_value * weight + bias)

// decimal points
WDATA_PREC = 5
X1DATA_PREC = 5
X2DATA_PREC = 5
XDATA_PREC = 5
module neuron #(
    parameter WDATA_WIDTH = 11 // synaptic weight
    parameter X1DATA_WIDTH = 14 // intermediate: w*x
    parameter X2DATA_WIDTH = 16  // summed w*x (concern: overflow)
    parameter XDATA_WIDTH = 10 // input and output
)(
    input signed [XDATA_WIDTH-1:0] input_value,
    input signed [WDATA_WIDTH-1:0] weight,
    input signed [WDATA_WIDTH-1:0] bias,
    output signed [XDATA_WIDTH-1:0] activated_output
);

    // was 2*XDATA_WIDTH
    wire signed [X1DATA_WIDTH-1:0] mult_result;
    wire signed [X2DATA_WIDTH-1:0] sum_result;

    wire signed [(WDATA_WIDTH+XDATA_WIDTH)-1:0] mult_result_raw;
    // wire signed [(X2DATA_WIDTH)-1:0] sum_result_raw;


    // assign mult_result = ( input_value * weight ) << D1;
    // assign sum_result = mult_result[X2DATA_WIDTH-1:0] + bias;

    // X → X1 → X2 → X
    XDATA_PREC + WDATA_PREC →→ X1DATA_PREC →→ X2DATA_PREC →→ XDATA_PREC
    D1 = X1DATA_PREC - (XDATA_PREC + WDATA_PREC)
    D2 = X2DATA_PREC - X1DATA_PREC
    D3 = XDATA_PREC - X2DATA_PREC
    assign mult_result_raw = input_value * weight;
    assign mult_result[X1DATA_WIDTH-1:0] = mult_result_raw[X1DATA_WIDTH-1+D1:D1] ; // << D1; // Ignore D1 bits
    assign sum_result[X2DATA_WIDTH-D2:D2] = mult_result + bias <<; // bias: shift D1+D2

    assign activated_output = (sum_result > 0) ? sum_result : 0;

endmodule
