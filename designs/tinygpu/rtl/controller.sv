`default_nettype none
`timescale 1ns/1ns
// DERIVED by designs/tinygpu/derive_rtl.py from upstream src/controller.sv
// (Didactic_GPU_Simulator, commit in designs/tinygpu/UPSTREAM_PIN). Do not edit by hand.
// MacroMind change: synthesis front-end fix only, behaviour-identical.
//   Upstream mixes blocking (=) and non-blocking (<=) writes to the serving-consumer
//   mask in one always block, which yosys-slang rejects (assert in procedural.cc).
//   Blocking claims now go to a blocking shadow `csc`, non-blocking releases to
//   `csc_clr`, and the register is written once with `csc & ~csc_clr`. NBAs land
//   after all blocking updates of a time step, so this equals the upstream semantics.

// Memory Controller
// Routes consumer (LSU/fetcher) requests onto a smaller pool of memory channels.
// When COALESCE = 1, sequential reads to identical addresses are merged: a single
// channel transaction broadcasts its result to every consumer that requested it.
// (Memory coalescing — Phase 5.)
module controller #(
    parameter ADDR_BITS       = 8,
    parameter DATA_BITS       = 16,
    parameter NUM_CONSUMERS   = 4,
    parameter NUM_CHANNELS    = 1,
    parameter WRITE_ENABLE    = 1,
    parameter COALESCE        = 0,
    parameter PRIORITY_ENABLE = 0
) (
    input  wire                          clk,
    input  wire                          reset,

    input  wire [NUM_CONSUMERS-1:0]      consumer_read_valid,
    input  wire [ADDR_BITS-1:0]          consumer_read_address  [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]      consumer_read_ready,
    output reg  [DATA_BITS-1:0]          consumer_read_data     [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]      consumer_write_valid,
    input  wire [ADDR_BITS-1:0]          consumer_write_address [NUM_CONSUMERS-1:0],
    input  wire [DATA_BITS-1:0]          consumer_write_data    [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]      consumer_write_ready,

    // Per-consumer priority bit. Reads are scanned first when PRIORITY_ENABLE=1.
    input  wire [NUM_CONSUMERS-1:0]      consumer_priority,

    output reg  [NUM_CHANNELS-1:0]       mem_read_valid,
    output reg  [ADDR_BITS-1:0]          mem_read_address       [NUM_CHANNELS-1:0],
    input  wire [NUM_CHANNELS-1:0]       mem_read_ready,
    input  wire [DATA_BITS-1:0]          mem_read_data          [NUM_CHANNELS-1:0],
    output reg  [NUM_CHANNELS-1:0]       mem_write_valid,
    output reg  [ADDR_BITS-1:0]          mem_write_address      [NUM_CHANNELS-1:0],
    output reg  [DATA_BITS-1:0]          mem_write_data         [NUM_CHANNELS-1:0],
    input  wire [NUM_CHANNELS-1:0]       mem_write_ready,

    // Telemetry
    output reg  [31:0]                   mem_requests,
    output reg  [31:0]                   mem_requests_coalesced
);
    localparam [2:0]
        IDLE           = 3'b000,
        READ_WAITING   = 3'b010,
        WRITE_WAITING  = 3'b011,
        READ_RELAYING  = 3'b100,
        WRITE_RELAYING = 3'b101;

    reg [2:0]                            controller_state [NUM_CHANNELS-1:0];
    reg [$clog2(NUM_CONSUMERS+1)-1:0]    current_consumer [NUM_CHANNELS-1:0];
    reg [NUM_CONSUMERS-1:0]              channel_serving_consumer;

    // Per-channel coalescing mask: which consumers are being served by this transaction.
    reg [NUM_CONSUMERS-1:0]              channel_coalesce_mask [NUM_CHANNELS-1:0];

    integer i, j;
    integer pass;
    reg     found;
    reg     can_consider;
    // Blocking accumulators: one count per transaction / merged consumer, across all channels in a cycle.
    reg [31:0] requests, coalesced;
    reg [NUM_CONSUMERS-1:0] csc, csc_clr;

    always @(posedge clk) begin
        if (reset) begin
            mem_read_valid       <= 0;
            mem_write_valid      <= 0;
            consumer_read_ready  <= 0;
            consumer_write_ready <= 0;
            channel_serving_consumer <= 0;
            mem_requests           <= 0;
            mem_requests_coalesced <= 0;
            for (i = 0; i < NUM_CHANNELS; i = i + 1) begin
                controller_state[i]      <= IDLE;
                current_consumer[i]      <= 0;
                mem_read_address[i]      <= 0;
                mem_write_address[i]     <= 0;
                mem_write_data[i]        <= 0;
                channel_coalesce_mask[i] <= 0;
            end
            for (i = 0; i < NUM_CONSUMERS; i = i + 1) consumer_read_data[i] <= 0;
        end else begin
            csc       = channel_serving_consumer;
            csc_clr   = {NUM_CONSUMERS{1'b0}};
            requests  = mem_requests;
            coalesced = mem_requests_coalesced;
            for (i = 0; i < NUM_CHANNELS; i = i + 1) begin
                case (controller_state[i])
                    IDLE: begin
                        found = 1'b0;
                        // Two-pass scan when priority is enabled: first only
                        // high-utility (priority) consumers, then everyone.
                        for (pass = 0; pass < (PRIORITY_ENABLE ? 2 : 1); pass = pass + 1) begin
                        for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                            can_consider = !PRIORITY_ENABLE || (pass == 1) || consumer_priority[j];
                            if (!found && can_consider && consumer_read_valid[j] && !csc[j]) begin
                                // Blocking claim (as upstream), so later channels this cycle skip j.
                                csc[j] = 1'b1;
                                current_consumer[i]         <= j[$clog2(NUM_CONSUMERS+1)-1:0];
                                mem_read_valid[i]           <= 1'b1;
                                mem_read_address[i]         <= consumer_read_address[j];
                                controller_state[i]         <= READ_WAITING;
                                requests                    = requests + 1;

                                // Coalescing pass: any other consumer reading the same address
                                // in the same cycle rides this transaction.
                                channel_coalesce_mask[i] <= ({{(NUM_CONSUMERS-1){1'b0}}, 1'b1} << j);
                                if (COALESCE) begin : coalesce_block
                                    integer k;
                                    for (k = 0; k < NUM_CONSUMERS; k = k + 1) begin
                                        if (k != j
                                            && consumer_read_valid[k]
                                            && !csc[k]
                                            && consumer_read_address[k] == consumer_read_address[j]) begin
                                            csc[k] = 1'b1;
                                            channel_coalesce_mask[i][k] <= 1'b1;
                                            coalesced = coalesced + 1;
                                        end
                                    end
                                end
                                found = 1'b1;
                            end
                            // Writes skip the priority pass, so one STR's lanes claim channels in lane order (ADR-016).
                            if (WRITE_ENABLE && !found && (!PRIORITY_ENABLE || pass == 1)
                                && consumer_write_valid[j] && !csc[j]) begin
                                csc[j] = 1'b1;
                                current_consumer[i]         <= j[$clog2(NUM_CONSUMERS+1)-1:0];
                                mem_write_valid[i]          <= 1'b1;
                                mem_write_address[i]        <= consumer_write_address[j];
                                mem_write_data[i]           <= consumer_write_data[j];
                                controller_state[i]         <= WRITE_WAITING;
                                requests                    = requests + 1;
                                channel_coalesce_mask[i]    <= ({{(NUM_CONSUMERS-1){1'b0}}, 1'b1} << j);
                                found = 1'b1;
                            end
                        end
                        end
                    end
                    READ_WAITING: if (mem_read_ready[i]) begin
                        mem_read_valid[i] <= 0;
                        // Broadcast to all coalesced consumers.
                        for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                            if (channel_coalesce_mask[i][j]) begin
                                consumer_read_ready[j] <= 1'b1;
                                consumer_read_data[j]  <= mem_read_data[i];
                            end
                        end
                        controller_state[i] <= READ_RELAYING;
                    end
                    WRITE_WAITING: if (mem_write_ready[i]) begin
                        mem_write_valid[i] <= 0;
                        consumer_write_ready[current_consumer[i]] <= 1'b1;
                        controller_state[i] <= WRITE_RELAYING;
                    end
                    READ_RELAYING: begin
                        // Wait until every coalesced consumer has dropped its valid.
                        if (~|(consumer_read_valid & channel_coalesce_mask[i])) begin
                            for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                                if (channel_coalesce_mask[i][j]) begin
                                    csc_clr[j] = 1'b1;
                                    consumer_read_ready[j]      <= 1'b0;
                                end
                            end
                            channel_coalesce_mask[i] <= 0;
                            controller_state[i]      <= IDLE;
                        end
                    end
                    WRITE_RELAYING: if (!consumer_write_valid[current_consumer[i]]) begin
                        csc_clr[current_consumer[i]] = 1'b1;
                        consumer_write_ready[current_consumer[i]]      <= 1'b0;
                        controller_state[i] <= IDLE;
                    end
                    default: ;
                endcase
            end
            channel_serving_consumer <= csc & ~csc_clr;
            mem_requests           <= requests;
            mem_requests_coalesced <= coalesced;
        end
    end
endmodule
