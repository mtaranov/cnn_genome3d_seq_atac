from __future__ import absolute_import, division, print_function

from builtins import zip
from genomedatalayer.extractors import (
    FastaExtractor, MemmappedBigwigExtractor, MemmappedFastaExtractor
)
import itertools
import numpy as np

def batch_iter(iterable, batch_size):
    '''iterates in batches.
    '''
    it = iter(iterable)
    try:
        while True:
            values = []
            for n in xrange(batch_size):
                values += (it.next(),)
            yield values
    except StopIteration:
        # yield remaining values
        yield values


def infinite_batch_iter(iterable, batch_size):
    '''iterates in batches indefinitely.
    '''
    return batch_iter(itertools.cycle(iterable),
                      batch_size)


def generate_from_intervals(intervals, extractors, batch_size=128, indefinitely=True):
    """
    Generates signals extracted on interval batches.

    Parameters
    ----------
    intervals : sequence of intervals
    extractors : list of gdl extractors
    batch_size : int, optional
    indefinitely : bool, default: True
    """
    interval_length = intervals[0].length
    # preallocate batch arrays
    batch_arrays = []
    for extractor in extractors:
        if type(extractor) in [FastaExtractor, MemmappedFastaExtractor]:
            batch_arrays.append(np.zeros((batch_size, 1, 4, interval_length), dtype=np.float32))
        elif type(extractor) in [MemmappedBigwigExtractor]:
            batch_arrays.append(np.zeros((batch_size, 1, 1, interval_length), dtype=np.float32))
    if indefinitely:
        batch_iterator = infinite_batch_iter(intervals, batch_size)
    else:
        batch_iterator = batch_iter(intervals, batch_size)
    for batch_intervals in batch_iterator:
        try:
            #yield [extractor(batch_intervals, out=batch_arrays[i]) for i, extractor in enumerate(extractors)]
            yield [extractor(batch_intervals) for i, extractor in enumerate(extractors)]
        except ValueError:
            yield [extractor(batch_intervals) for extractor in extractors]


def generate_from_interval_pairs(intervals1, intervals2, extractors, batch_size=128, indefinitely=True):
    """
    Zips two instances of generate_from_intervals and stacks their outputs
    """
    interval_length = intervals1[0].length
    batch_arrays = []
    for extractor in extractors:
        if type(extractor) in [FastaExtractor, MemmappedFastaExtractor]:
            batch_arrays.append(np.zeros((batch_size, 1, 8, interval_length), dtype=np.float32))
        elif type(extractor) in [MemmappedBigwigExtractor]:
            batch_arrays.append(np.zeros((batch_size, 1, 2, interval_length), dtype=np.float32))
    batch_generator = zip(generate_from_intervals(intervals1, extractors,
                                                  batch_size=batch_size,
                                                  indefinitely=indefinitely),
                          generate_from_intervals(intervals2, extractors,
                                                  batch_size=batch_size,
                                                  indefinitely=indefinitely))
    for (inputs1, inputs2) in batch_generator:
        try:
            for (input1, input2, batch_array) in zip(inputs1, inputs2, batch_arrays):
                print(input1.shape)
                batch_array[: , :, :input1.shape[-2], :] = input1
                batch_array[: , :, input2.shape[-2]:, :] = input2
            yield batch_arrays
        except ValueError:
            last_batch_arrays = []
            for (input1, input2) in zip(inputs1, inputs2):
                last_batch_arrays.append(np.concatenate((input1, input2), axis=-2))
            yield last_batch_arrays


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def generate_from_array(array, batch_size=128, indefinitely=True):
    """
    Generates the array in batches.
    """
    if indefinitely:
        batch_iterator = infinite_batch_iter(array, batch_size)
    else:
        batch_iterator = batch_iter(array, batch_size)
    for array_batch in batch_iterator:
        yield np.stack(array_batch, axis=0)


def generate_from_interval_pairs_and_labels(intervals1, intervals2, labels, extractors, batch_size=128, indefinitely=True):
    """
    Generates batches of (inputs1, inputs2, labels) where inputs is a list of numpy arrays based on provided extractors.
    """
    batch_generator = zip(generate_from_interval_pairs(intervals1, intervals2, extractors,
                                                       batch_size=batch_size, indefinitely=indefinitely),
                          generate_from_array(labels, batch_size=batch_size, indefinitely=indefinitely))
    for batch in batch_generator:
        yield batch
    
