// Simulation-only behavioural model of the OpenRAM macro sky130_sram_1rw1r_64x256_8
// (ORFS platforms/sky130ram), same ports and cycle behaviour as the vendor model
// minus its per-access $display: inputs registered at posedge; write at negedge
// (byte mask); port-1 read data valid DELAY after negedge and X in any cycle
// without a read, so a wrapper that does not hold read data fails in simulation.
`timescale 1ns/1ps
module sky130_sram_1rw1r_64x256_8 (
    input  wire        clk0, input wire csb0, input wire web0, input wire [7:0] wmask0,
    input  wire [7:0]  addr0, input wire [63:0] din0, output reg [63:0] dout0,
    input  wire        clk1, input wire csb1, input wire [7:0] addr1, output reg [63:0] dout1
);
    localparam DELAY = 1;
    reg [63:0] mem [0:255];
    reg csb0_r, web0_r, csb1_r; reg [7:0] wmask0_r, addr0_r, addr1_r; reg [63:0] din0_r;
    integer i;
    always @(posedge clk0) begin
        csb0_r <= csb0; web0_r <= web0; wmask0_r <= wmask0; addr0_r <= addr0; din0_r <= din0;
        dout0 <= 64'bx;
    end
    always @(posedge clk1) begin
        csb1_r <= csb1; addr1_r <= addr1;
        dout1 <= 64'bx;
    end
    always @(negedge clk0)
        if (!csb0_r && !web0_r)
            for (i = 0; i < 8; i = i + 1)
                if (wmask0_r[i]) mem[addr0_r][i*8 +: 8] = din0_r[i*8 +: 8];
    always @(negedge clk1)
        if (!csb1_r) dout1 <= #(DELAY) mem[addr1_r];
endmodule
