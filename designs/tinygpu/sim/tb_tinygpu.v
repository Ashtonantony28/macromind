// MacroMind: self-contained Icarus testbench for tinygpu_top (one kernel launch).
// Plusargs: +prog=<hex> +data=<hex> +out=<hex> +threads=<n>  (hex files: one word per line)
// Memory model: zero-latency, like upstream test/helpers/memory.py with MEM_LATENCY=0
// (ready/data answer a valid in the following cycle).
`timescale 1ns/1ns
module tb;
    localparam DCH = `TG_DATA_CHANNELS, AW = `TG_DATA_ADDR_BITS, WB = `TG_WORD_BITS;
    reg clk = 0, reset = 1, start = 0, dcr_we = 0;
    reg [7:0] dcr_data = 0;
    wire done;
    wire p_valid; wire [7:0] p_addr; reg p_ready = 0; reg [15:0] p_data = 0;
    wire [DCH-1:0] r_valid, w_valid;
    wire [DCH*AW-1:0] r_addr, w_addr;
    wire [DCH*WB-1:0] w_data;
    reg  [DCH-1:0] r_ready = 0, w_ready = 0;
    reg  [DCH*WB-1:0] r_data = 0;

    tinygpu_top dut (
        .clk(clk), .reset(reset), .start(start), .done(done),
        .device_control_write_enable(dcr_we), .device_control_data(dcr_data),
        .program_mem_read_valid(p_valid), .program_mem_read_address(p_addr),
        .program_mem_read_ready(p_ready), .program_mem_read_data(p_data),
        .data_mem_read_valid(r_valid), .data_mem_read_address(r_addr),
        .data_mem_read_ready(r_ready), .data_mem_read_data(r_data),
        .data_mem_write_valid(w_valid), .data_mem_write_address(w_addr),
        .data_mem_write_data(w_data), .data_mem_write_ready(w_ready));

    reg [15:0] pmem [0:255];
    reg [WB-1:0] dmem [0:(1<<AW)-1];
    integer i, cycles, threads, fd, maxc, nfetch, nrd, nwr;
    reg [1023:0] f;
    always #5 clk = ~clk;

    initial begin nfetch = 0; nrd = 0; nwr = 0; end
    always @(posedge clk) begin
        if (p_valid && !p_ready) nfetch = nfetch + 1;
        for (i = 0; i < DCH; i = i + 1) begin
            if (r_valid[i] && !r_ready[i]) nrd = nrd + 1;
            if (w_valid[i] && !w_ready[i]) nwr = nwr + 1;
        end
    end
    always @(posedge clk) begin
        p_ready <= p_valid && !p_ready;
        p_data  <= pmem[p_addr];
        for (i = 0; i < DCH; i = i + 1) begin
            r_ready[i] <= r_valid[i] && !r_ready[i];
            r_data[i*WB +: WB] <= dmem[r_addr[i*AW +: AW]];
            w_ready[i] <= w_valid[i] && !w_ready[i];
            if (w_valid[i] && !w_ready[i]) dmem[w_addr[i*AW +: AW]] <= w_data[i*WB +: WB];
        end
    end

    initial begin
        for (i = 0; i < 256; i = i + 1) pmem[i] = 0;
        for (i = 0; i < (1<<AW); i = i + 1) dmem[i] = 0;
        if ($value$plusargs("prog=%s", f)) $readmemh(f, pmem);
        if ($value$plusargs("data=%s", f)) $readmemh(f, dmem);
        if (!$value$plusargs("threads=%d", threads)) threads = 4;
        if (!$value$plusargs("maxcycles=%d", maxc)) maxc = 2000000;
        repeat (3) @(posedge clk);
        reset <= 0;
        @(posedge clk); dcr_we <= 1; dcr_data <= threads[7:0];
        @(posedge clk); dcr_we <= 0; start <= 1;
        cycles = 0;
        while (!done && cycles < maxc) begin @(posedge clk); cycles = cycles + 1; end
        $display("TB_DONE done=%0d cycles=%0d fetches=%0d reads=%0d writes=%0d", done, cycles, nfetch, nrd, nwr);
        if ($value$plusargs("out=%s", f)) begin
            fd = $fopen(f, "w");
            for (i = 0; i < (1<<AW); i = i + 1) $fdisplay(fd, "%h", dmem[i]);
            $fclose(fd);
        end
        $finish;
    end
endmodule
