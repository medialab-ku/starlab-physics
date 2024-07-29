import math

import taichi as ti
@ti.data_oriented
class BlellochScan:
    def __init__(self, num_array):
        self.num_array = num_array
        self.num_steps = int(math.log2(self.num_array))
        self.prefix_sum_temp = ti.field(ti.i32, shape=self.num_array)

    @ti.kernel
    def upsweep(self, prefix_sum: ti.template(), step: ti.int32, size: ti.int32):
        offset = step - 1

        ti.loop_config(block_dim=64)
        for i in range(size):
            id = offset + step * i
            prefix_sum[id] += prefix_sum[id - (step >> 1)]

    @ti.kernel
    def downsweep(self, prefix_sum: ti.template(), step: ti.int32, size: ti.int32):
        offset = step - 1
        offset_rev = (step >> 1)

        ti.loop_config(block_dim=64)
        for i in range(size):
            id = offset + step * i
            temp = prefix_sum[id - offset_rev]
            prefix_sum[id - offset_rev] = prefix_sum[id]
            prefix_sum[id] += temp

    @ti.kernel
    def add_count(self, prefix_sum: ti.template()):
        for i in range(self.num_array):
            prefix_sum[i] += self.prefix_sum_temp[i]

    def run(self, prefix_sum: ti.template()):

        self.prefix_sum_temp.copy_from(prefix_sum)
        d = 0
        test = self.num_array
        while test > 1:
            step = 1 << (d + 1)
            size = self.num_array // step
            self.upsweep(prefix_sum, step, size)

            d += 1
            test //= 2

        prefix_sum[self.num_array - 1] = 0
        d = self.num_steps - 1

        while d >= 0:
            step = 1 << (d + 1)
            size = self.num_array // step
            self.downsweep(prefix_sum, step, size)
            d -= 1

        self.add_count(prefix_sum)
