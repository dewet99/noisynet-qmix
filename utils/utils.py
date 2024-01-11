import torch as th
from typing import Tuple, Union
import math
import random
import operator
from typing import Callable

class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32
    
class RunningMeanStdTorch(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = th.zeros(shape, dtype=th.float32, device=device)
        self.var = th.ones(shape, dtype=th.float32, device=device)
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = th.mean(arr, dim=0)
        batch_var = th.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + th.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class SumTree(object):
	def __init__(self, max_size):
		self.max_size = max_size
		self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
		self.tree_size = 2 ** self.tree_level - 1
		self.tree = [0. for _ in range(self.tree_size)]
		self.size = 0
		self.cursor = 0

	def add(self, value):
		index = self.cursor
		self.cursor = (self.cursor + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		self.val_update(index, value)

	def get_val(self, index):
		tree_index = 2 ** (self.tree_level - 1) - 1 + index
		return self.tree[tree_index]

	def val_update(self, index, value):
		tree_index = 2 ** (self.tree_level - 1) - 1 + index
		diff = value - self.tree[tree_index]
		self.reconstruct(tree_index, diff)

	def reconstruct(self, tindex, diff):
		self.tree[tindex] += diff
		if not tindex == 0:
			tindex = int((tindex - 1) / 2)
			self.reconstruct(tindex, diff)

	def find(self, value, norm=True):
		pre_value = value
		# print(f"Tree[0]: {self.tree[0]}")
		if norm:
			value *= self.tree[0]
		list = []
		# print(f"recursive find starts now as find({value}, 0, {pre_value}, {list})")
		return self._find(value, 0, pre_value, list)

	def _find(self, value, index, r, list):
		# print(f"recurse find ({value}, {index}, {r}, {list})")
		if 2 ** (self.tree_level - 1) - 1 <= index:
			if index - (2 ** (self.tree_level - 1) - 1) >= self.size:
				print('!!!!!')
				print("Index, value, self.tree[0], r")
				print(index, value, self.tree[0], r)
				print(list)
				index = (2 ** (self.tree_level - 1) - 1) + random.randint(0, self.size)
				#index = (2 ** (self.tree_level - 1) - 1)
			return self.tree[index], index - (2 ** (self.tree_level - 1) - 1)

		left = self.tree[2 * index + 1]
		list.append(left)

		if value <= left + 1e-8:
			return self._find(value, 2 * index + 1, r, list)
		else:
			return self._find(value - left, 2 * (index + 1), r, list)

	def print_tree(self):
		for k in range(1, self.tree_level + 1):
			for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
				print(self.tree[j])
			print()

	def filled_size(self):
		return self.size


class Experience(object):
	""" The class represents prioritized experience replay buffer.

	The class has functions: store samples, pick samples with
	probability in proportion to sample's priority, update
	each sample's priority, reset alpha.

	see https://arxiv.org/pdf/1511.05952.pdf .

	"""

	def __init__(self, memory_size, alpha=1):
		self.tree = SumTree(memory_size)
		self.memory_size = memory_size
		self.alpha = alpha

	def add(self, priority):
		self.tree.add(priority ** self.alpha)

	def select(self, batch_size):

		if self.tree.filled_size() < batch_size:
			return None

		indices = []
		priorities = []
		for _ in range(batch_size):
			r = random.random()
			# print(f"Finding value of r: {r} in tree")
			priority, index = self.tree.find(r)
			if index == self.memory_size:
				while index == self.memory_size:
					r = random.random()
					priority, index = self.tree.find(r)
			priorities.append(priority)
			indices.append(index)
			self.priority_update([index], [0])  # To avoid duplicating

		self.priority_update(indices, priorities)  # Revert priorities

		return indices

	def priority_update(self, indices, priorities):
		""" The methods update samples's priority.

		Parameters
		----------
		indices :
			list of sample indices
		"""
		for i, p in zip(indices, priorities):
			self.tree.val_update(i, p ** self.alpha)

def conv_output_shape(
	h_w: Tuple[int, int],
	kernel_size: Union[int, Tuple[int, int]] = 1,
	stride: int = 1,
	padding: int = 0,
	dilation: int = 1,
	) -> Tuple[int, int]:
		"""
		Calculates the output shape (height and width) of the output of a convolution layer.
		kernel_size, stride, padding and dilation correspond to the inputs of the
		torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
		:param h_w: The height and width of the input.
		:param kernel_size: The size of the kernel of the convolution (can be an int or a
		tuple [width, height])
		:param stride: The stride of the convolution
		:param padding: The padding of the convolution
		:param dilation: The dilation of the convolution
		"""
		from math import floor

		if not isinstance(kernel_size, tuple):
			kernel_size = (int(kernel_size), int(kernel_size))

		h = floor(
			((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
		)
		w = floor(
			((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
		)
		return h, w

def signed_hyperbolic(x: th.Tensor, eps: float = 1e-3) -> th.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    # base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return th.sign(x) * (th.sqrt(th.abs(x) + 1) - 1) + eps * x

def signed_parabolic(x: th.Tensor, eps: float = 1e-3) -> th.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    # base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    z = th.sqrt(1 + 4 * eps * (eps + 1 + th.abs(x))) / 2 / eps - 1 / 2 / eps
    return th.sign(x) * (th.square(z) - 1)


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)