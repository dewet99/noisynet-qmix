
import torch as th
import numpy as np
from types import SimpleNamespace as SN
import pdb
import time
import numpy as np

import ray
from utils.utils import OneHot, Experience
import traceback
# from . import PER_EpisodeBatch

class EpisodeBatch(object):

    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float16)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):

        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):

        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        # print(f"bs: {bs}")
        # print(f"ts: {ts}")
        slices = self._parse_slices((bs, ts))
        # print(f"post parse slices: {slices}")
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
        
            if not isinstance(v, th.Tensor):
                if isinstance(v, list):
                    v = np.array(v)
                v = th.tensor(v, dtype=dtype, device=self.device)

            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])
            

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        # print("============================================")
        # print("_check_safe_view called")
        # print(f"v: {v}")
        # print(f"dest: {dest}")
        # print("============================================")
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        # print("============================================")
        # print("_check_safe_view called")
        # print(f"item: {item}")
        
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)

            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            # print(f"returns: {ret}")
            # print("============================================")
            return ret

    def _get_num_items(self, indexing_item, max_size):
        # print("============================================")
        # print(f"Get num items called:")
        # print(f"indexing_item: {indexing_item}")
        # print(f"max_size: {max_size}")
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())

    def slice_gru_experiences_random_starts(self, max_t_filled, recurrent_sequence_length):
        """
        Slices a nmber of sequential steps from each episode in the batch.
        The starting index of each episode is different.
        This is very slow because you need to loop through everything that exists to be able
        to get the data. 
        Parameters:
        max_t_filled: integer tensor of shape ([1])
        recurrent_sequence_length: int
        """
        # Create a new episodeBatch to contain the sliced data, so we don't need
        # to modify the original replay buffer.
        # Do it like done in q_learner -> duplicate batch
        sgt = time.time()
        # Scheme is a tiny dict, copy should be fast
        scheme = {k: v for k, v in self.scheme.items() if k != "filled"}

        new_batch = EpisodeBatch(scheme,
                                 self.groups,
                                 self.batch_size,
                                 recurrent_sequence_length,
                                 )
        data = {}
        # loop thru all episodes in sample
        for b in range(self.batch_size):
            # loop through all items in the batch
            start_idx = np.random.randint(0, max_t_filled-recurrent_sequence_length+1)
            
            # print(f"slicing from {start_idx[0]} to {start_idx[0]+recurrent_sequence_length} in batch {b}")
            for k,v in self.data.transition_data.items():
                # If the entry k has not yet been made in the dictionairy, create it
                if str(k) not in data.keys():
                    # start_idx is an array of length 1
                    data[k] = v[b, start_idx[0]:start_idx[0]+recurrent_sequence_length, ...].unsqueeze(0)
                # if k (which is like obs, actions, state, etc) already exists, stack the new value 
                else:
                    data[k] = th.cat([data[k], v[b, start_idx[0]:start_idx[0]+recurrent_sequence_length, ...].unsqueeze(0)], dim = 0)
        # after all this, new_batch should have same stuff as original batch, except that
        # every episode has had 64 values sliced from it
        new_batch.update(data=data)
        print(f"Slice gru experiences takes {(time.time()- sgt)*1000}ms for batch size : {self.batch_size}")
        
        return new_batch
    
    def slice_gru_experiences_same_starts(self, max_t_filled, recurrent_sequence_length):
        """
        Slices a nmber of sequential steps from each episode in the batch.
        The starting index of each episode is the same
        Parameters:
        max_t_filled: integer tensor of shape ([1])
        recurrent_sequence_length: int
        """
         # Create a new episodeBatch to contain the sliced data, so we don't need
        # to modify the original replay buffer.
        # Do it like done in q_learner -> duplicate batch
        sgt = time.time()
        # Scheme is a tiny dict, copy should be fast
        scheme = {k: v for k, v in self.scheme.items() if k != "filled"}

        new_batch = EpisodeBatch(scheme,
                                 self.groups,
                                 self.batch_size,
                                 recurrent_sequence_length,
                                 )
        data = {}
        # loop through all items in the batch
        start_idx = np.random.randint(0, max_t_filled-recurrent_sequence_length+1)
        
        # print(f"slicing from {start_idx[0]} to {start_idx[0]+recurrent_sequence_length} in batch {b}")
        for k,v in self.data.transition_data.items():
                # start_idx is an array of length 1
            data[k] = v[:, start_idx[0]:start_idx[0]+recurrent_sequence_length, ...]
           # after all this, new_batch should have same stuff as original batch, except that
        # every episode has had 64 values sliced from it
        new_batch.update(data=data)
        print(f"Slice gru experiences takes {(time.time()- sgt)*1000}ms for batch size : {self.batch_size}")
        
        return new_batch


class Remote_EpisodeBatch(object):
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float16)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):

        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):

        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        # print(f"bs: {bs}")
        # print(f"ts: {ts}")
        slices = self._parse_slices((bs, ts))
        # print(f"post parse slices: {slices}")
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
        
            if not isinstance(v, th.Tensor):
                v = th.tensor(v, dtype=dtype, device=self.device)

            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])
            

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        # print("============================================")
        # print("_check_safe_view called")
        # print(f"v: {v}")
        # print(f"dest: {dest}")
        # print("============================================")
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        # print("============================================")
        # print("_check_safe_view called")
        # print(f"item: {item}")
        
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = Remote_EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)

            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = Remote_EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            # print(f"returns: {ret}")
            # print("============================================")
            return ret

    def _get_num_items(self, indexing_item, max_size):
        # print("============================================")
        # print(f"Get num items called:")
        # print(f"indexing_item: {indexing_item}")
        # print(f"max_size: {max_size}")
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    # def __repr__(self):
    #     return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
    #                                                                                  self.max_seq_length,
    #                                                                                  self.scheme.keys(),
    #                                                                                  self.groups.keys())


@ray.remote(num_cpus=1)
class Remote_ReplayBuffer(Remote_EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        # pdb.set_trace()
        
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:

            buffer_left = self.buffer_size - self.buffer_index

            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

  

    def can_sample(self, batch_size):
        """
        Returns true if there are more episodes present in the buffer than the batch size
        """
        return self.episodes_in_buffer > batch_size
    
    def is_full(self):
        return self.episodes_in_buffer==self.buffer_size 

    def sample(self, batch_size):
        assert self.can_sample(batch_size), "Assert can_sample(batch_size) failed"
        # if self.episodes_in_buffer == batch_size:
        #     # pdb.set_trace()
        #     print("returning all episodes")
        #     print(self[:batch_size])
        #     print("Printed tore")
        #     return self[:batch_size]
        # else:
            # Uniform sampling only atm
        
        ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        toret = self[ep_ids]

        return toret

    def get_scheme(self):
        return self.scheme

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


@ray.remote(num_cpus=1)
class Prioritized_ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, alpha, preprocess=None, device="cpu"):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.proportional = Experience(buffer_size, alpha=alpha)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        try:
            for i in range(ep_batch.batch_size):
                self.proportional.add(100)
            if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
                self.update(ep_batch.data.transition_data,
                            slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                            slice(0, ep_batch.max_seq_length),
                            mark_filled=False)
                self.update(ep_batch.data.episode_data,
                            slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
                self.buffer_index = (self.buffer_index + ep_batch.batch_size)
                self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
                self.buffer_index = self.buffer_index % self.buffer_size
                assert self.buffer_index < self.buffer_size
            else:
                buffer_left = self.buffer_size - self.buffer_index
                self.insert_episode_batch(ep_batch[0:buffer_left, :])
                self.insert_episode_batch(ep_batch[buffer_left:, :])
        except Exception as e:
            traceback.print_exc()

    def can_sample(self, batch_size):
        return self.episodes_in_buffer > batch_size

    def sample(self, batch_size, newest=False):
        assert self.can_sample(batch_size)
        if newest and self.episodes_in_buffer >= batch_size:
            return self[self.episodes_in_buffer - batch_size: self.episodes_in_buffer]
        elif self.episodes_in_buffer == batch_size:
            return np.arange(batch_size), self[:batch_size]
        else:
            # Uniform sampling only atm
            
            ep_ids = self.proportional.select(batch_size)
            # if self.buffer_size in ep_ids:
            #     while self.buffer_index in ep_ids:
            #         ep_ids = self.proportional.select(batch_size)

            assert (ep_ids != None)
            # ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            # print(ep_ids)
            # print(ep_ids.shape)
            try:
                return ep_ids, self[ep_ids]
            except Exception as e:
                print(f"Selected epi_ids: {ep_ids}")
                traceback.print_exc()
    def update_priority(self, indices, priorities):
        self.proportional.priority_update(indices, priorities)

@ray.remote(num_cpus=1)
class CustomPrioritized_ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, alpha, preprocess=None, device="cpu"):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        # self.proportional = Experience(buffer_size, alpha=alpha)
        self.alpha=alpha
        self.buffer_size = buffer_size  # Max number of episodes in batch
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.lowest_priority = 1e-6
        self._priorities = np.ones((buffer_size,), dtype = np.float32)
        self._random_state = np.random.RandomState()
        self._max_seen_priority = 1.0

        # try:
        #     # Tree stuff
        #     tree_capacity = 1
        #     while tree_capacity < self.buffer_size:
        #         tree_capacity*=2

        #     self.sum_tree = SumSegmentTree(tree_capacity)
        # except Exception as e:
        #     traceback.print_exc()
        # # self.min_tree = MinSegmentTree(tree_capacity)

    def insert_episode_batch(self, ep_batch):
        try:
            # for i in range(ep_batch.batch_size):
            #     self.proportional.add(100)
            if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
                self.update(ep_batch.data.transition_data,
                            slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                            slice(0, ep_batch.max_seq_length),
                            mark_filled=False)
                self.update(ep_batch.data.episode_data,
                            slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
                self.buffer_index = (self.buffer_index + ep_batch.batch_size)
                self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
                self.buffer_index = self.buffer_index % self.buffer_size
                self._priorities[self.buffer_index] = self._max_seen_priority
                assert self.buffer_index < self.buffer_size
            else:
                buffer_left = self.buffer_size - self.buffer_index
                self.insert_episode_batch(ep_batch[0:buffer_left, :])
                self.insert_episode_batch(ep_batch[buffer_left:, :])
                self._priorities[self.buffer_index] = self._max_seen_priority
            # try:
            #     self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            #     # self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            #     self.tree_ptr = (self.tree_ptr+1)%self.buffer_size
            # except Exception as e:
            #     traceback.print_exc()

        except Exception as e:
            traceback.print_exc()

    def can_sample(self, batch_size):
        return self.episodes_in_buffer > batch_size

    def sample(self, batch_size, newest=False):
        try:
            assert self.can_sample(batch_size), "Not enough eps"
            # if newest and self.episodes_in_buffer >= batch_size:
            #     return self[self.episodes_in_buffer - batch_size: self.episodes_in_buffer]
            # elif self.episodes_in_buffer == batch_size:
            #     return np.arange(batch_size), self[:batch_size]
            # else:
            # Uniform sampling only atm
            try:
                ep_ids = self._sample_proportional(batch_size)
            except Exception as e:
                traceback.print_exc()
            # if self.buffer_size in ep_ids:
            #     while self.buffer_index in ep_ids:
            #         ep_ids = self.proportional.select(batch_size)

            assert (ep_ids != None)
            # ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            # print(ep_ids)
            # print(ep_ids.shape)
            try:
                return ep_ids, self[ep_ids]
            except Exception as e:
                print(f"Selected epi_ids: {ep_ids}")
                traceback.print_exc()
        except Exception as e:
            traceback.print_exc()

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority, np.max(priorities)])


        for i in range(priorities.shape[0]):
            if not priorities[i] > 0:
                priorities[i] = self.lowest_priority
                print(f"We have a priority of {priorities[i]}, setting to {self.lowest_priority}")


        self._priorities[indices] = priorities.squeeze()
        # try:
        #     assert len(indices) == len(priorities), "Number of indices and number of priorities aren't the same"

        #     for idx, priority in zip(indices, priorities):
        #         # assert priority>0, f"Priority is not greater than 0, it contains {priorities}"
        #         if not priority > 0:
        #             # We get a nan priority sometimes, rather than fix the source of the bug like a good programmer, which is
        #             # that the neural networks output some weiiird values sometimes, we just assign that transition a very, very low priority
                    
        #             priority = self.lowest_priority
        #         assert 0 <= idx < self.episodes_in_buffer, "Index is greater than the number of episodes in the buffer or less than zero"
        #         print(f"prioritys: {priority**self.alpha}")
        #         self.sum_tree[idx] = priority**self.alpha
        #         # self.min_tree[idx] = priority**self.alpha

        #         self.max_priority = max(self.max_priority, priority)
        # except Exception as e:
        #     traceback.print_exc()


    def _sample_proportional(self, batch_size):
        # try:
        #     indices = []
        #     p_total = self.sum_tree.sum(0, self.episodes_in_buffer-1) #This may need to change
        #     print(f"p_total is: {p_total}")
        #     segment = p_total/self.batch_size

        #     for i in range(batch_size):
        #         a = segment*i
        #         b = segment * (i+1)
        #         upperbound = random.uniform(a,b)
        #         idx = self.sum_tree.retrieve(upperbound)
        #         indices.append(idx)
        # except Exception as e:
        #     traceback.print_exc()
        # print("========================")
        # print("In sample_proportional")
        priorities = self._priorities[:self.episodes_in_buffer] ** self.alpha
        # print(f"Priorites: {priorities}")
        probs = priorities/np.sum(priorities)
        # print(f"probs: {probs.shape}")
        indices = self._random_state.choice(np.arange(probs.shape[0]), size = batch_size, replace=True, p = probs)
        # print(f"indices: {indices}")
        # print("========================")
        
        return list(indices)


def generate_replay_scheme(config):

    scheme = {
            "state": {"vshape": config["state_shape"]},
            "obs": {"vshape": config["obs_shape"], "group": "agents", "dtype": th.uint8},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.uint8},
            "avail_actions": {"vshape": (config["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
        
    if config["curiosity"]:
        icm_reward = {"icm_reward": {"vshape": (1,)},}
        scheme.update(icm_reward)

    if config["use_burnin"]:
        hidden_states = {"hidden_state": {"vshape": (1, 2,config["rnn_hidden_dim"]), "dtype": th.float32}}
        scheme.update(hidden_states)

    
    groups = {
    "agents": config["num_agents"]
    }

    preprocess = {
    "actions": ("actions_onehot", [OneHot(out_dim=config["n_actions"])])
    }

    return scheme, groups, preprocess
