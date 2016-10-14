#!/usr/bin/python

r'''more-reasonable core functionality for numpy

* SYNOPSIS
    >>> import numpy as np
    >>> import numpysane as nps

    >>> a   = np.arange(6).reshape(2,3)
    >>> b   = a + 100
    >>> row = a[0,:]

    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> b
    array([[100, 101, 102],
           [103, 104, 105]])

    >>> row
    array([1000, 1001, 1002])

    >>> nps.glue(a,b, axis=-1)
    array([[  0,   1,   2, 100, 101, 102],
           [  3,   4,   5, 103, 104, 105]])

    >>> nps.glue(a,b,row, axis=-2)
    array([[   0,    1,    2],
           [   3,    4,    5],
           [ 100,  101,  102],
           [ 103,  104,  105],
           [1000, 1001, 1002]])

    >>> nps.cat(a,b)
    array([[[  0,   1,   2],
            [  3,   4,   5]],

           [[100, 101, 102],
            [103, 104, 105]]])

    >>> @nps.broadcast_define( (('n',), ('n',)) )
    ... def inner_product(a, b):
    ...     return a.dot(b)

    >>> inner_product(a,b)
    array([ 305, 1250])

* DESCRIPTION
Numpy is widely used, relatively polished, and has a wide range of libraries
available. At the same time, some of its very core functionality is strange,
confusing and just plain wrong. This is in contrast with PDL
(http://pdl.perl.org), which has a much more reasonable core, but a number of
higher-level warts, and a relative dearth of library support. This module
intends to improve the developer experience by providing alternate APIs to some
core numpy functionality that is much more reasonable, especially for those who
have used PDL in the past.

Instead of writing a new module (this module), it would be really nice to simply
patch numpy to give everybody the more reasonable behavior. I'd be very happy to
do that, but the issues lie with some very core functionality, and any changes
in behavior would likely break existing code. Any comments in how to achieve
better behaviors in a less forky manner as welcome.

Finally, if the system DOES make sense in some way that I'm simply not
understanding, I'm happy to listen. I have no intention to disparage anyone or
anything; I just want a more usable system for numerical computations.

The issues addressed by this module fall into two broad categories:

1. Incomplete broadcasting support
2. Strange, special-case-ridden rules for basic array manipulation, especially
   dealing with dimensionality

** Broadcasting
*** Problem
Numpy has a limited support for broadcasting
(http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), a generic way
to vectorize functions. When making a broadcasted call to a function, you pass
in arguments with the inputs to vectorize available in new dimensions, and the
broadcasting mechanism automatically calls the function multiple times as
needed, and reports the output as an array collecting all the results.

A basic example is an inner product: a function that takes in two
identically-sized vectors (1-dimensional arrays) and returns a scalar
(0-dimensional array). A broadcasted inner product function could take in two
arrays of shape (2,3,4), compute the 6 inner products of length-4 each, and
report the output in an array of shape (2,3). Numpy puts the most-significant
dimension at the end, which is why this isn't 12 inner products of length-2
each. This is a semi-arbitrary design choice, which could have been made
differently: PDL puts the most-significant dimension at the front, for instance.

The user doesn't choose whether to use broadcasting or not: some functions
support it, and some do not. In PDL, broadcasting (called "threading" in that
system) is a pervasive concept throughout. A PDL user has an expectation that
every function can broadcast, and the documentation for every function is very
explicit about the dimensionality of the inputs and outputs. Any data above the
expected input dimensions is broadcast.

By contrast, in numpy very few functions know how to broadcast. On top of that,
the documentation is usually silent about the broadcasting status of a function
in question. And on top of THAT, broadcasting rules state that an array of
dimensions (n,m) is functionally identical to one of dimensions
(1,1,1,....1,n,m). However, many numpy functions have special-case rules to
create different behaviors for inputs with different numbers of dimensions, and
this creates unexpected results. The effect of all this is a messy situation
where the user is often not sure of the exact behavior of the functions they're
calling, and trial and error is required to make the system do what one wants.

*** Solution
This module contains functionality to make any arbitrary function broadcastable.
This is invoked as a decorator, applied to the arbitrary user function. An
example:

    >>> import numpysane as nps

    >>> @nps.broadcast_define( (('n',), ('n',)) )
    ... def inner_product(a, b):
    ...     return a.dot(b)

Here we have a simple inner product function to compute ONE inner product. We
call 'broadcast_define' to add a broadcasting-aware wrapper that takes two 1D
vectors of length 'n' each (same 'n' for the two inputs). This new
'inner_product' function applies broadcasting, as needed:

    >>> import numpy as np

    >>> a = np.arange(6).reshape(2,3)
    >>> b = a + 100

    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> b
    array([[100, 101, 102],
           [103, 104, 105]])

    >>> inner_product(a,b)
    array([ 305, 1250])

A detailed description of broadcasting rules is available in the numpy
documentation: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

In short:

- The most significant dimension in a numpy array is the LAST one, so the
  prototype of an input argument must exactly match a given input's trailing
  shape. So a prototype shape of (a,b,c) accepts an argument shape of (......,
  a,b,c), with as many or as few leading dimensions as desired.
- The extra leading dimensions must be compatible across all the inputs. This
  means that each leading dimension must either
  - equal to 1
  - be missing (thus assumed to equal 1)
  - equal to some positive integer >1, consistent across all arguments
- The output is collected into an array that's sized as a superset of the
  above-prototype shape of each argument

More involved example: A function with input prototype ( (3,), ('n',3), ('n',),
('m',) ) given inputs of shape

    (1,5,    3)
    (2,1,  8,3)
    (        8)
    (  5,    9)

will return an output array of shape (2,5, ...), where ... is the shape of each
output slice. Note again that the prototype dictates the TRAILING shape of the
inputs.

Another related function in this module broadcast_generate(). It's similar to
broadcast_define(), but instead of adding broadcasting-awareness to an existing
function, it simply generates tuples from a set of arguments according to a
given prototype.

Stock numpy has some rudimentary support for all this with its vectorize()
function, but it assumes only scalar inputs and outputs, which severaly limits
its usefulness.

*** New planned functionality

In addition to this basic broadcasting support, I'm planning the following:

- A C-level broadcast_define(). This would be the analogue of PDL::PP
  (http://pdl.perl.org/PDLdocs/PP.html). This flavor of broadcast_define() would
  be invoked by the build system to wrap C functions. It would implement
  broadcasting awareness in C code it generates, which should work more
  effectively for performance-sensitive inner loops.

- Automatic parallelization for broadcasted slices. Since each broadcasting loop
  is independent, this is a very natural place to add parallelism.

- Dimensions should support a symbolic declaration. For instance, one could want
  a function to accept an input of shape (n) and another of shape (n*n). There's
  no way to declare this currently, but there should be.

** Strangeness in core routines
*** Problem
There are some core numpy functions whose behavior is strange, full of special
cases and very confusing, at least to me. That makes it difficult to achieve
some very basic things. In the following examples, I use a function "arr" that
returns a numpy array with given dimensions:

    >>> def arr(*shape):
    ...     product = reduce( lambda x,y: x*y, shape)
    ...     return np.arange(product).reshape(*shape)

    >>> arr(1,2,3)
    array([[[0, 1, 2],
            [3, 4, 5]]])

    >>> arr(1,2,3).shape
    (1, 2, 3)

The following sections are an incomplete list of the strange functionality I've
encountered.

**** Concatenation
A prime example of confusing functionality is the array concatenation routines.
Numpy has a number of functions to do this, each being strange.

***** hstack()
hstack() performs a "horizontal" concatenation. When numpy prints an array, this
is the last dimension (remember, the most significant dimensions in numpy are at
the end). So one would expect that this function concatenates arrays along this
last dimension. In the special case of 1D and 2D arrays, one would be right:

    >>> np.hstack( (arr(3), arr(3))).shape
    (6,)

    >>> np.hstack( (arr(2,3), arr(2,3))).shape
    (2, 6)

but in any other case, one would be wrong:

    >>> np.hstack( (arr(1,2,3), arr(1,2,3))).shape
    (1, 4, 3)     <------ I expect (1, 2, 6)

    >>> np.hstack( (arr(1,2,3), arr(1,2,4))).shape
    [exception]   <------ I expect (1, 2, 7)

    >>> np.hstack( (arr(3), arr(1,3))).shape
    [exception]   <------ I expect (1, 6)

    >>> np.hstack( (arr(1,3), arr(3))).shape
    [exception]   <------ I expect (1, 6)

I think the above should all succeed, and should produce the shapes as
indicated. Cases such as "np.hstack( (arr(3), arr(1,3)))" are maybe up for
debate, but broadcasting rules allow adding as many extra length-1 dimensions as
we want without changing the meaning of the object, so I claim this should work.
Either way, if you print out the operands for any of the above, you too would
expect a "horizontal" stack() to work as stated above.

It turns out that normally hstack() concatenates along axis=1, unless the first
argument only has one dimension, in which case axis=0 is used. This is 100%
wrong in a system where the most significant dimension is the last one, unless
you assume that everyone has only 2D arrays, where the last dimension and the
second dimension are the same.

The correct way to do this is to concatenate along axis=-1. It works for
n-dimensionsal objects, and doesn't require the special case logic for
1-dimensional objects that hstack() has.

***** vstack()
Similarly, vstack() performs a "vertical" concatenation. When numpy prints an
array, this is the second-to-last dimension (remember, the most significant
dimensions in numpy are at the end). So one would expect that this function
concatenates arrays along this second-to-last dimension. In the special
case of 1D and 2D arrays, one would be right:

    >>> np.vstack( (arr(2,3), arr(2,3))).shape
    (4, 3)

    >>> np.vstack( (arr(3), arr(3))).shape
    (2, 3)

    >>> np.vstack( (arr(1,3), arr(3))).shape
    (2, 3)

    >>> np.vstack( (arr(3), arr(1,3))).shape
    (2, 3)

    >>> np.vstack( (arr(2,3), arr(3))).shape
    (3, 3)

Note that this function appears to tolerate some amount of shape mismatches. It
does it in a form one would expect, but given the state of the rest of this
system, I found it surprising. For instance "np.hstack( (arr(1,3), arr(3)))"
fails, so one would think that "np.vstack( (arr(1,3), arr(3)))" would fail too.

And once again, adding more dimensions make it confused, for the same reason:

    >>> np.vstack( (arr(1,2,3), arr(2,3))).shape
    [exception]   <------ I expect (1, 4, 3)

    >>> np.vstack( (arr(1,2,3), arr(1,2,3))).shape
    (2, 2, 3)     <------ I expect (1, 4, 3)

Similarly to hstack(), vstack() concatenates along axis=0, which is "vertical"
only for 2D arrays, but not for any others. And similarly to hstack(), the 1D
case has special-cased logic to work properly.

The correct way to do this is to concatenate along axis=-2. It works for
n-dimensionsal objects, and doesn't require the special case for 1-dimensional
objects that vstack() has.

***** dstack()
I'll skip the detailed description, since this is similar to hstack() and
vstack(). The intent was to concatenate across axis=-3, but the implementation
takes axis=2 instead. This is wrong, as before. And I find it strange that these
3 functions even exist, since they are all special-cases: the concatenation axis
should be an argument, and at most, the edge special case (hstack()) should
exist. This brings us to the next function:

***** concatenate()
This is a more general function, and unlike hstack(), vstack() and dstack(), it
takes as input a list of arrays AND the concatenation dimension. It accepts
negative concatenation dimensions to allow us to count from the end, so things
should work better. And in many ways that failed previously, they do:

    >>> np.concatenate( (arr(1,2,3), arr(1,2,3)), axis=-1).shape
    (1, 2, 6)

    >>> np.concatenate( (arr(1,2,3), arr(1,2,4)), axis=-1).shape
    (1, 2, 7)

    >>> np.concatenate( (arr(1,2,3), arr(1,2,3)), axis=-2).shape
    (1, 4, 3)

But many things still don't work as I would expect:

    >>> np.concatenate( (arr(1,3), arr(3)), axis=-1).shape
    [exception]   <------ I expect (1, 6)

    >>> np.concatenate( (arr(3), arr(1,3)), axis=-1).shape
    [exception]   <------ I expect (1, 6)

    >>> np.concatenate( (arr(1,3), arr(3)), axis=-2).shape
    [exception]   <------ I expect (3, 3)

    >>> np.concatenate( (arr(3), arr(1,3)), axis=-2).shape
    [exception]   <------ I expect (2, 3)

    >>> np.concatenate( (arr(2,3), arr(2,3)), axis=-3).shape
    [exception]   <------ I expect (2, 2, 3)

This function works as expected only if

- All inputs have the same number of dimensions
- All inputs have a matching shape, except for the dimension along which we're
  concatenating
- All inputs HAVE the dimension along which we're concatenating

A legitimate use case that violates these conditions: I have an object that
contains N 3D vectors, and I want to add another 3D vector to it. This is
essentially the first failing example above.

***** stack()
The name makes it sound exactly like concatenate(), and it takes the same
arguments, but it is very different. stack() requires that all inputs have
EXACTLY the same shape. It then concatenates all the inputs along a new
dimension, and places that dimension in the location given by the 'axis' input.
If this is the exact type of concatenation you want, this function works fine.
But it's one of many things a user may want to do.

**** inner() and dot()
Another arbitrary example of a strange API is np.dot() and np.inner(). In a
real-valued n-dimensional Euclidean space, a "dot product" is just another name
for an "inner product". Numpy disagrees.

It looks like np.dot() is matrix multiplication, with some wonky behaviors when
given higher-dimension objects, and with some special-case behaviors for
1-dimensional and 0-dimensional objects:

    >>> np.dot( arr(4,5,2,3), arr(3,5)).shape
    (4, 5, 2, 5) <--- expected result for a broadcasted matrix multiplication

    >>> np.dot( arr(3,5), arr(4,5,2,3)).shape
    [exception] <--- np.dot() is not commutative.
                     Expected for matrix multiplication, but not for a dot
                     product

    >>> np.dot( arr(4,5,2,3), arr(1,3,5)).shape
    (4, 5, 2, 1, 5) <--- don't know where this came from at all

    >>> np.dot( arr(4,5,2,3), arr(3)).shape
    (4, 5, 2) <--- 1D special case. This is a dot product.

    >>> np.dot( arr(4,5,2,3), 3).shape
    (4, 5, 2, 3) <--- 0D special case. This is a scaling.

It looks like np.inner() is some sort of quasi-broadcastable inner product, also
with some funny dimensioning rules. In many cases it looks like np.dot(a,b) is
the same as np.inner(a, transpose(b)) where transpose() swaps the last two
dimensions:


    >>> np.inner( arr(4,5,2,3), arr(5,3)).shape
    (4, 5, 2, 5) <--- All the length-3 inner products collected into a shape
                      with not-quite-broadcasting rules

    >>> np.inner( arr(5,3), arr(4,5,2,3)).shape
    (5, 4, 5, 2) <--- np.inner() is not commutative. Unexpected
                      for an inner product

    >>> np.inner( arr(4,5,2,3), arr(1,5,3)).shape
    (4, 5, 2, 1, 5) <--- No idea

    >>> np.inner( arr(4,5,2,3), arr(3)).shape
    (4, 5, 2) <--- 1D special case. This is a dot product.

    >>> np.inner( arr(4,5,2,3), 3).shape
    (4, 5, 2, 3) <--- 0D special case. This is a scaling.

**** atleast_xd()
Numpy has 3 special-case functions atleast_1d(), atleast_2d() and atleast_3d().
For 4d and higher, you need to do something else. As expected by now, these do
surprising things:

    >>> np.atleast_3d( arr(3)).shape
    (1, 3, 1)

I don't know when this is what I would want, so we move on.


*** Solution
This module introduces new functions that can be used for this core
functionality instead of the builtin numpy functions. These new functions work
in ways that (I think) are more intuitive and more reasonable. They do not refer
to anything being "horizontal" or "vertical", nor do they talk about "rows" or
"columns"; these concepts simply don't apply in a generic N-dimensional system.
These functions are very explicit about the dimensionality of the
inputs/outputs, and fit well into a broadcasting-aware system. Furthermore, the
names and semantics of these new functions come directly from PDL, which is more
consistent in this area.

Since these functions assume that broadcasting is an important concept in the
system, the given axis indices should be counted from the most significant
dimension: the last dimension in numpy. This means that where an axis index is
specified, negative indices are encouraged. glue() forbids axis>=0 outright.


Example for further justification:

An array containing N 3D vectors would have shape (N,3). Another array
containing a single 3D vector would have shape (3). Counting the dimensions from
the end, each vector is indexed in dimension -1. However, counting from the
front, the vector is indexed in dimension 0 or 1, depending on which of the two
arrays we're looking at. If we want to add the single vector to the array
containing the N vectors, and we mistakenly try to concatenate along the first
dimension, it would fail if N != 3. But if we're unlucky, and N=3, then we'd get
a nonsensical output array of shape (3,4). Why would an array of N 3D vectors
have shape (N,3) and not (3,N)? Because if we apply python iteration to it, we'd
expect to get N iterates of arrays with shape (3,) each, and numpy iterates from
the first dimension:

    >>> a = np.arange(2*3).reshape(2,3)

    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> [x for x in a]
    [array([0, 1, 2]), array([3, 4, 5])]

New functions this module provides (documented fully in the next section):

**** glue
Concatenates arrays along a given axis. Implicit length-1 dimensions are added
at the start as needed. Dimensions other than the glueing axis must match
exactly.

**** cat
Concatenate a given list of arrays along a new least-significant (leading) axis.
Again, implicit length-1 dimensions are added, and the resulting shapes must
match, and no data duplication occurs.

**** clump
Reshapes the array by grouping together the 'n' most significant dimensions,
where 'n' is given. So for instance, if x.shape is (2,3,4) then
nps.clump(x,2).shape is (2,12)

**** atleast_dims
Adds length-1 dimensions at the front of an array so that all the given
dimensions are in-bounds. Given axis<0 can expand the shape; given axis>=0 MUST
already be in-bounds. This preserves the alignment of the most-significant axis
index.

**** mv
Moves a dimension from one position to another

**** xchg
Exchanges the positions of two dimensions

**** transpose
Reverses the order of the two most significant dimensions in an array. The whole
array is seen as being an array of 2D matrices, each matrix living in the 2 most
significant dimensions, which implies this definition.

**** dummy
Adds a single length-1 dimension at the given position

**** reorder
Completely reorders the dimensions in an array

**** dot
Broadcast-aware non-conjugating dot product. Identical to inner

**** vdot
Broadcast-aware conjugating dot product

**** inner
Broadcast-aware inner product. Identical to dot

**** outer
Broadcast-aware outer product.

**** matmult
Broadcast-aware matrix multiplication

*** New planned functionality
The functions listed above are a start, but more will be added with time.

'''

import numpy as np
from functools import reduce
import types
import inspect


# object needed for fancy slices. m[:] is exactly the same as
# m[_colon], but '_colon' can be manipulated in ways that ':' can't
_colon = slice(None, None, None)

def _product(l):
    r'''Returns product of all values in the given list'''
    return reduce( lambda a,b: a*b, l )


def _clone_function(f, name):
    r'''Returns a clone of a given function.

    This is useful to copy a function, updating its metadata, such as the
    documentation, name, etc. There are also differences here between python 2
    and python 3 that this function handles.

    '''
    def get(f, what):
        what2 = 'func_{}'.format(what)
        what3 = '__{}__' .format(what)
        try:
            return getattr(f, what2)
        except:
            try:
                return getattr(f, what3)
            except:
                pass
        return None

    return types.FunctionType(get(f, 'code'),
                              get(f, 'globals'),
                              name,
                              get(f, 'defaults'),
                              get(f, 'closure'))



class NumpysaneError(Exception):
    def __init__(self, err): self.err = err
    def __str__(self):       return self.err


def _eval_broadcast_dims( args, prototype, dims_extra, dims_named ):
    r'''Helper function to evaluate a given list of arguments in respect to a given
    broadcasting prototype. This function will flag any errors in the
    dimensionality of the inputs. If no errors are detected, the outer shape of
    the broadcast is stored in the dims_extra list, and the values of the
    symbolic dimensions are stored in the dims_named dict

    '''

    def parse_dim( name_arg,
                   shape_prototype, shape_arg, dims_extra, dims_named ):

        def range_rev(n):
            r'''Returns a range from -1 to -n.

            Useful to index variable-sized lists while aligning their ends.'''
            return [-i-1 for i in range(n)]



        # first, I make sure the input is at least as dimension-ful as the
        # prototype. I make this a hard requirement. Even if the missing
        # dimensions have length 1, it is likely a mistake on the part of the
        # user
        if len(shape_prototype) > len(shape_arg):
            raise NumpysaneError("Argument {} has {} dimensions, but the prototype {} demands at least {}".format(name_arg, len(shape_arg), shape_prototype, len(shape_prototype)))

        # Loop through the dimensions. Set the dimensionality of any new named
        # argument to whatever the current argument has. Any already-known
        # argument must match
        for i_dim in range_rev(len(shape_prototype)):
            if not isinstance(shape_prototype[i_dim], int) and \
               shape_prototype[i_dim] not in dims_named:
                dims_named[shape_prototype[i_dim]] = shape_arg[i_dim]

            # The prototype dimension (named or otherwise) now has a numeric
            # value. Make sure it matches what I have
            dim_prototype = dims_named[shape_prototype[i_dim]] \
                            if not isinstance(shape_prototype[i_dim], int) \
                               else shape_prototype[i_dim]

            if dim_prototype != shape_arg[i_dim]:
                raise NumpysaneError("Argument {} dimension '{}': expected {} but got {}".
                    format(name_arg,
                           shape_prototype[i_dim],
                           dim_prototype,
                           shape_arg[i_dim]))

        # I now know that this argument matches the prototype. I look at the
        # extra dimensions to broadcast, and make sure they match with the
        # dimensions I saw previously
        ndims_extra_here = len(shape_arg) - len(shape_prototype)

        # This argument has ndims_extra_here dimensions to broadcast. The
        # current shape to broadcast must be at least as large, and must match
        if ndims_extra_here > len(dims_extra):
            dims_extra[:0] = [1] * (ndims_extra_here - len(dims_extra))
        for i_dim in range_rev(ndims_extra_here):
            dim_arg = shape_arg[i_dim - len(shape_prototype)]
            if dim_arg != 1:
                if dims_extra[i_dim] == 1:
                    dims_extra[i_dim] = dim_arg
                elif dims_extra[i_dim] != dim_arg:
                    raise NumpysaneError("Argument {} prototype {} extra broadcast dim {} mismatch: previous arg set this to {}, but this arg wants {}".
                        format(name_arg,
                               shape_prototype,
                               i_dim,
                               dims_extra[i_dim],
                               dim_arg))


    # clear out the accumulators
    dims_extra[:] = []
    dims_named.clear()
    for i_arg in range(len(args)):
        parse_dim( i_arg,
                   prototype[i_arg], args[i_arg].shape,
                   dims_extra, dims_named )



def _broadcast_iter_dim( i_dims_extra,

                         idx_slices, args, prototype,
                         dims_extra ):
    r'''Recursive generator to iterate through all the broadcasting slices.

    Each recursive call loops through a single dimension. I can do
    some of this with itertools.product(), and maybe using that
    would be a better choice.

    i_dims_extra is an integer indexing the current extra dimension
    we're looking at.

    idx_slices is an array of indices for each argument that is
    filled in by this function. This may vary for each argument
    because of varying prototypes and varying broadcasting shapes.

    '''

    if idx_slices is None:
        idx_slices = [[0]*(x.ndim-len(p)) + [_colon]*len(p) for p,x in zip(prototype,args)]

    if i_dims_extra < len(dims_extra):
        # more dimensions remaining. recurse

        # idx_slices contains the indices for each argument.
        # Two notes:
        #
        # 1. Any indices indexing above the first dimension should be
        #    omitted
        # 2. Indices into a higher dimension of length 1 should be left at 0
        for dim in range(dims_extra[i_dims_extra]):
            for x,p,idx_slice in zip(args,prototype,idx_slices):
                x_dim = x.ndim - (len(p) + len(dims_extra) - i_dims_extra)
                if x_dim >= 0 and x.shape[x_dim] > 1:
                    idx_slice[x_dim] = dim

            for x in _broadcast_iter_dim( i_dims_extra+1,

                                          idx_slices, args, prototype,
                                          dims_extra):
                yield x
        return

    # This is the last dimension. Evaluate this slice.

    # the "if idx else x" business if for 0-dimensional arrays.
    # tuple(idx) because of wonky behavior differences:
    #     >>> a
    #     array([[0, 1, 2],
    #            [3, 4, 5]])
    #
    #     >>> a[tuple((1,1))]
    #     4
    #
    #     >>> a[list((1,1))]
    #     array([[3, 4, 5],
    #            [3, 4, 5]])
    yield tuple( x[tuple(idx)] if idx else x for idx,x in zip(idx_slices, args) )




def broadcast_define(prototype, prototype_output=None, out_kwarg=None):
    r'''Vectorizes an arbitrary function, expecting input as in the given prototype.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> @nps.broadcast_define( (('n',), ('n',)) )
        ... def inner_product(a, b):
        ...     return a.dot(b)

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> inner_product(a,b)
        array([ 305, 1250])


    The prototype defines the dimensionality of the inputs. In the inner product
    example above, the input is two 1D n-dimensional vectors. In particular, the
    'n' is the same for the two inputs. This function is intended to be used as
    a decorator, applied to a function defining the operation to be vectorized.
    Each element in the prototype list refers to each input, in order. In turn,
    each such element is a list that describes the shape of that input. Each of
    these shape descriptors can be any of

    - a positive integer, indicating an input dimension of exactly that length
    - a string, indicating an arbitrary, but internally consistent dimension

    The normal numpy broadcasting rules (as described elsewhere) apply. In
    summary:

    - Dimensions are aligned at the end of the shape list, and must match the
      prototype

    - Extra dimensions left over at the front must be consistent for all the
      input arguments, meaning:

      - All dimensions !=1 must be identical
      - Missing dimensions are implicitly set to 1

    - The output has a shape where
      - The trailing dimensions are whatever the function being broadcasted
        outputs
      - The leading dimensions come from the extra dimensions in the inputs

    Scalars are represented as 0-dimensional numpy arrays: arrays with shape (),
    and these broadcast as one would expect:

        >>> @nps.broadcast_define( (('n',), ('n',), ()))
        ... def scaled_inner_product(a, b, scale):
        ...     return a.dot(b)*scale

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> scale = np.array((10,100))

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> scale
        array([ 10, 100])

        >>> scaled_inner_product(a,b,scale)
        array([[  3050],
               [125000]])

    Let's look at a more involved example. Let's say we have a function that
    takes a set of points in R^2 and a single center point in R^2, and finds a
    best-fit least-squares line that passes through the given center point. Let
    it return a 3D vector containing the slope, y-intercept and the RMS residual
    of the fit. This broadcasting-enabled function can be defined like this:

        import numpy as np
        import numpysane as nps

        @nps.broadcast_define( (('n',2), (2,)) )
        def fit(xy, c):
            # line-through-origin-model: y = m*x
            # E = sum( (m*x - y)**2 )
            # dE/dm = 2*sum( (m*x-y)*x ) = 0
            # ----> m = sum(x*y)/sum(x*x)
            x,y = (xy - c).transpose()
            m = np.sum(x*y) / np.sum(x*x)
            err = m*x - y
            err **= 2
            rms = np.sqrt(err.mean())
            # I return m,b because I need to translate the line back
            b = c[1] - m*c[0]

            return np.array((m,b,rms))

    And I can use broadcasting to compute a number of these fits at once. Let's
    say I want to compute 4 different fits of 5 points each. I can do this:

        n = 5
        m = 4
        c = np.array((20,300))
        xy = np.arange(m*n*2, dtype=np.float64).reshape(m,n,2) + c
        xy += np.random.rand(*xy.shape)*5

        res = fit( xy, c )
        mb  = res[..., 0:2]
        rms = res[..., 2]
        print "RMS residuals: {}".format(rms)

    Here I had 4 different sets of points, but a single center point c. If I
    wanted 4 different center points, I could pass c as an array of shape (4,2).
    I can use broadcasting to plot all the results (the points and the fitted
    lines):

        import gnuplotlib as gp

        gp.plot( *nps.mv(xy,-1,0), _with='linespoints',
                 equation=['{}*x + {}'.format(mb_single[0],
                                              mb_single[1]) for mb_single in mb],
                 unset='grid', square=1)

    The examples above all create a separate output array for each broadcasted
    slice, and copy the contents from each such slice into the large output
    array that contains all the results. This is inefficient, and it is possible
    to pre-allocate an array to forgo these extra allocations and copies. There
    are several settings to control this. If the function being broadcasted can
    write its output to a given array instead of creating a new one, most of the
    inefficiency goes away. broadcast_define() supports the case where this
    function takes this array in a kwarg: the name of this kwarg can be given to
    broadcast_define() like so:

        @nps.broadcast_define( ....., out_kwarg = "out" )
        def func( ....., out):
            .....
            out[:] = result

    In order for broadcast_define() to pass such an output array to the inner
    function, this output array must be available, which means that it must be
    given to us somehow, or we must create it.

    The most efficient way to make a broadcasted call is to create the full
    output array beforehand, and to pass that to the broadcasted function. In
    this case, nothing extra will be allocated, and no unnecessary copies will
    be made. This can be done like this:

        @nps.broadcast_define( (('n',), ('n',)), ....., out_kwarg = "out" )
        def inner_product(a, b, out):
            .....
            out.setfield(a.dot(b), out.dtype)
            return out

        out = np.empty((2,4), float)
        inner_product( np.arange(3), np.arange(2*4*3).reshape(2,4,3), out=out)

    In this example, the caller knows that it's calling an inner_product
    function, and that the shape of each output slice would be (). The caller
    also knows the input dimensions and that we have an extra broadcasting
    dimension (2,4), so the output array will have shape (2,4) + () = (2,4).
    With this knowledge, the caller preallocates the array, and passes it to the
    broadcasted function call. Furthermore, in this case the inner function will
    be called with an output array EVERY time, and this is the only mode the
    inner function needs to support.

    If the caller doesn't know (or doesn't want to pre-compute) the shape of the
    output, it can let the broadcasting machinery create this array for them. In
    order for this to be possible, the shape of the output should be
    pre-declared, and the dtype of the output should be known:

        @nps.broadcast_define( (('n',), ('n',)),
                               (),
                               out_kwarg = "out" )
        def inner_product(a, b, out):
            .....
            out.setfield(a.dot(b), out.dtype)
            return out

        out = inner_product( np.arange(3), np.arange(2*4*3).reshape(2,4,3), dtype=int)

    Note that the caller didn't need to specify the prototype of the output or
    the extra broadcasting dimensions (output prototype is in the
    broadcast_define() call, but not the inner_product() call). Specifying the
    dtype here is optional: it defaults to float if omitted. If we want the
    output array to be pre-allocated, the output prototype (it is () in this
    example) is required: we must know the shape of the output array in order to
    create it.

    Without a declared output prototype, we can still make mostly- efficient
    calls: the broadcasting mechanism can call the inner function for the first
    slice as we showed earlier, by creating a new array for the slice. This new
    array required an extra allocation and copy, but it contains the required
    shape information. This infomation will be used to allocate the output, and
    the subsequent calls to the inner function will be efficient:

        @nps.broadcast_define( (('n',), ('n',)),
                               out_kwarg = "out" )
        def inner_product(a, b, out=None):
            .....
            if out is None:
                return a.dot(b)
            out.setfield(a.dot(b), out.dtype)
            return out

        out = inner_product( np.arange(3), np.arange(2*4*3).reshape(2,4,3))

    Here we were slighly inefficient, but the ONLY required extra specification
    was out_kwarg: that's mostly all you need. Also it is important to note that
    in this case the inner function is called both with passing it an output
    array to fill in, and with asking it to create a new one (by passing
    out=None to the inner function). This inner function then must support both
    modes of operation. If the inner function does not support filling in an
    output array, none of these efficiency improvements are possible.

    broadcast_define() is analogous to thread_define() in PDL.

    '''
    def inner_decorator_for_some_reason(func):
        # args broadcast, kwargs do not. All auxillary data should go into the
        # kwargs
        def broadcast_loop(*args, **kwargs):

            if len(args) < len(prototype):
                raise NumpysaneError("Mismatched number of input arguments. Wanted at least {} but got {}". \
                                      format(len(prototype), len(args)))

            args_passthru = args[  len(prototype):]
            args          = args[0:len(prototype) ]

            # make sure all the arguments are numpy arrays
            args = tuple(np.asarray(arg) for arg in args)

            dims_extra = [] # extra dimensions to broadcast through
            dims_named = {} # values of the named dimensions
            _eval_broadcast_dims( args, prototype, dims_extra, dims_named )

            # if no broadcasting involved, just call the function
            if not dims_extra:
                sliced_args = args + args_passthru
                return func( *sliced_args, **kwargs )

            # I checked all the dimensions and aligned everything. I have my
            # to-broadcast dimension counts. Iterate through all the broadcasting
            # output, and gather the results
            output = None

            # reshaped output. I write to this array
            output_flattened = None

            # substitute named variable values into the output prototype
            prototype_output_local = None
            if prototype_output is not None:
                prototype_output_local = [d if type(d) is int
                                          else dims_named[d] for d in prototype_output]

            # if the output was supposed to go to a particular place, set that
            kwargs_dtype = {}
            if 'dtype' in kwargs:
                kwargs_dtype['dtype'] = kwargs['dtype']
            if out_kwarg and out_kwarg in kwargs:
                output = kwargs[out_kwarg]
                if prototype_output_local is not None:
                    expected_shape = dims_extra + prototype_output_local
                    if output.shape != tuple(expected_shape):
                        raise NumpysaneError("Inconsistent output shape: expected {}, but got {}".format(expected_shape, output.shape))
            # if we know enough to allocate the output, do that
            elif prototype_output_local is not None:
                output = np.empty(dims_extra + prototype_output_local,
                                  **kwargs_dtype)

            if output is not None:
                output_flattened = output.reshape( (_product(dims_extra),) +
                                                   output.shape[len(dims_extra):] )

            i_slice = 0
            for x in _broadcast_iter_dim( 0,
                                          None, args, prototype,
                                          dims_extra ):

                # if the function knows how to write directly to an array,
                # request that
                if output is not None and out_kwarg:
                    kwargs[out_kwarg] = output_flattened[i_slice, ...]

                sliced_args = x + args_passthru
                result = func( *sliced_args, **kwargs )

                if output is None:
                    output = np.empty( dims_extra + list(result.shape),
                                       dtype = result.dtype)
                    output_flattened = output.reshape( (_product(dims_extra),) + result.shape)

                    output_flattened[i_slice, ...] = result
                elif not out_kwarg:
                    output_flattened[i_slice, ...] = result

                if prototype_output_local is None:
                    prototype_output_local = result.shape
                elif result.shape != tuple(prototype_output_local):
                    raise NumpysaneError("Inconsistent slice output shape: expected {}, but got {}".format(prototype_output_local, result.shape))

                i_slice = i_slice+1

            return output


        func_out = _clone_function( broadcast_loop, func.__name__ )
        func_out.__doc__  = inspect.getdoc(func)
        if func_out.__doc__ is None:
            func_out.__doc__ = ''
        func_out.__doc__+= \
'''\n\nThis function is broadcast-aware through numpysane.broadcast_define().
The expected inputs have input prototype:

    {prototype}

{output_prototype_text}

The first {nargs} positional arguments will broadcast. The trailing shape of
those arguments must match the input prototype; the leading shape must follow
the standard broadcasting rules. Positional arguments past the first {nargs} and
all the keyword arguments are passed through untouched.'''. \
        format(prototype = prototype,
               output_prototype_text = 'No output prototype is defined.' if prototype_output is None else
               'and output prototype\n\n    {}'.format(prototype_output),
               nargs = len(prototype))
        return func_out
    return inner_decorator_for_some_reason


def broadcast_generate(prototype, args):
    r'''A generator that produces broadcasted slices

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> for s in nps.broadcast_generate( (('n',), ('n',)), (a,b)):
        ...     print "slice: {}".format(s)
        slice: (array([0, 1, 2]), array([100, 101, 102]))
        slice: (array([3, 4, 5]), array([103, 104, 105]))
    '''

    if len(args) != len(prototype):
        raise NumpysaneError("Mismatched number of input arguments. Wanted {} but got {}". \
                              format(len(prototype), len(args)))

    # make sure all the arguments are numpy arrays
    args = tuple(np.asarray(arg) for arg in args)

    dims_extra = [] # extra dimensions to broadcast through
    _eval_broadcast_dims( args, prototype, dims_extra, {} )

    # I checked all the dimensions and aligned everything. I have my
    # to-broadcast dimension counts. Iterate through all the broadcasting
    # output, and gather the results
    for x in _broadcast_iter_dim( 0,
                                  None, args, prototype,
                                  dims_extra ):
        yield x


def glue(*args, **kwargs):
    r'''Concatenates a given list of arrays along the given 'axis' keyword argument.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> row = a[0,:] + 1000

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> row
        array([1000, 1001, 1002])

        >>> nps.glue(a,b, axis=-1)
        array([[  0,   1,   2, 100, 101, 102],
               [  3,   4,   5, 103, 104, 105]])

        >>> nps.glue(a,b,row, axis=-2)
        array([[   0,    1,    2],
               [   3,    4,    5],
               [ 100,  101,  102],
               [ 103,  104,  105],
               [1000, 1001, 1002]])

        >>> nps.glue(a,b, axis=-3)
        array([[[  0,   1,   2],
                [  3,   4,   5]],

               [[100, 101, 102],
                [103, 104, 105]]])

    If no 'axis' keyword argument is given, a new dimension is added at the
    front, and we concatenate along that new dimension. This case is equivalent
    to numpysane.cat()

    In order to count dimensions from the inner-most outwards, this function accepts
    only negative axis arguments. This is because numpy broadcasts from the last
    dimension, and the last dimension is the inner-most in the (usual) internal
    storage scheme. Allowing glue() to look at dimensions at the start would allow
    it to unalign the broadcasting dimensions, which is never what you want.

    To glue along the last dimension, pass axis=-1; to glue along the second-to-last
    dimension, pass axis=-2, and so on.

    Unlike in PDL, this function refuses to create duplicated data to make the
    shapes fit. In my experience, this isn't what you want, and can create bugs.
    For instance, PDL does this:

        pdl> p sequence(3,2)
        [
         [0 1 2]
         [3 4 5]
        ]

        pdl> p sequence(3)
        [0 1 2]

        pdl> p PDL::glue( 0, sequence(3,2), sequence(3) )
        [
         [0 1 2 0 1 2]   <--- Note the duplicated "0,1,2"
         [3 4 5 0 1 2]
        ]

    while numpysane.glue() does this:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a[0:1,:]


        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[0, 1, 2]])

        >>> nps.glue(a,b,axis=-1)
        [exception]

    Finally, this function adds as many length-1 dimensions at the front as
    required. Note that this does not create new data, just new degenerate
    dimensions. Example:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> res = nps.glue(a,b, axis=-5)
        >>> res
        array([[[[[  0,   1,   2],
                  [  3,   4,   5]]]],



               [[[[100, 101, 102],
                  [103, 104, 105]]]]])

        >>> res.shape
        (2, 1, 1, 2, 3)

    '''

    axis = kwargs.get('axis')
    if axis is not None and axis >= 0:
        raise NumpysaneError("axis >= 0 can make broadcasting dimensions inconsistent, and is thus not allowed")

    # deal with scalar (non-ndarray) args
    args = [ np.asarray(x) for x in args ]

    # If no axis is given, add a new axis at the front, and glue along it
    max_ndim = max( x.ndim for x in args )
    if axis is None:
        axis = -1 - max_ndim

    # if we're glueing along a dimension beyond what we already have, expand the
    # target dimension count
    if max_ndim < -axis:
        max_ndim = -axis

    # Now I add dummy dimensions at the front of each array, to bring the source
    # arrays to the same dimensionality. After this is done, ndims for all the
    # matrices will be the same, and np.concatenate() should know what to do.
    args = [ x[(np.newaxis,)*(max_ndim - x.ndim) + (Ellipsis,)] for x in args ]

    return np.concatenate( args, axis=axis )


def cat(*args):
    r'''Concatenates a given list of arrays along a new first (outer) dimension.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> c = a - 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> c
        array([[-100,  -99,  -98],
               [ -97,  -96,  -95]])

        >>> res = nps.cat(a,b,c)
        >>> res
        array([[[   0,    1,    2],
                [   3,    4,    5]],

               [[ 100,  101,  102],
                [ 103,  104,  105]],

               [[-100,  -99,  -98],
                [ -97,  -96,  -95]]])

        >>> res.shape
        (3, 2, 3)

        >>> [x for x in res]
        [array([[0, 1, 2],
                [3, 4, 5]]),
         array([[100, 101, 102],
                [103, 104, 105]]),
         array([[-100,  -99,  -98],
                [ -97,  -96,  -95]])]

    This function concatenates the input arrays into an array shaped like the
    highest-dimensioned input, but with a new outer (at the start) dimension.
    The concatenation axis is this new dimension.

    As usual, the dimensions are aligned along the last one, so broadcasting
    will continue to work as expected. Note that this is the opposite operation
    from iterating a numpy array; see the example above.

    '''
    return glue(*args) # axis is unspecified


def clump(x, **kwargs):
    r'''Groups the given n most significant dimensions together.

    Synopsis:

        >>> import numpysane as nps
        >>> nps.clump( arr(2,3,4), n=2).shape
        (2, 12)
    '''
    n = kwargs.get('n')
    if n is None:
        raise NumpysaneError("clump() requires a dimension count in the 'n' kwarg")
    if n < 0:
        raise NumpysaneError("clump() requires n > 0")
    if n <= 1:
        return x

    if x.ndim < n:
        n = x.ndim

    s = list(x.shape[:-n]) + [ _product(x.shape[-n:]) ]
    return x.reshape(s)

def atleast_dims(x, *dims):
    r'''Returns an array with extra length-1 dimensions to contain all given axes.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> nps.atleast_dims(a, -1).shape
        (2, 3)

        >>> nps.atleast_dims(a, -2).shape
        (2, 3)

        >>> nps.atleast_dims(a, -3).shape
        (1, 2, 3)

        >>> nps.atleast_dims(a, 0).shape
        (2, 3)

        >>> nps.atleast_dims(a, 1).shape
        (2, 3)

        >>> nps.atleast_dims(a, 2).shape
        [exception]

        >>> l = [-3,-2,-1,0,1]
        >>> nps.atleast_dims(a, l).shape
        (1, 2, 3)

        >>> l
        [-3, -2, -1, 1, 2]

    If the given axes already exist in the given array, the given array itself
    is returned. Otherwise length-1 dimensions are added to the front until all
    the requested dimensions exist. The given axis>=0 dimensions MUST all be
    in-bounds from the start, otherwise the most-significant axis becomes
    unaligned; an exception is thrown if this is violated. The given axis<0
    dimensions that are out-of-bounds result in new dimensions added at the
    front.

    If new dimensions need to be added at the front, then any axis>=0 indices
    become offset. For instance:

        >>> x.shape
        (2, 3, 4)

        >>> [x.shape[i] for i in (0,-1)]
        [2, 4]

        >>> x = nps.atleast_dims(x, 0, -1, -5)
        >>> x.shape
        (1, 1, 2, 3, 4)

        >>> [x.shape[i] for i in (0,-1)]
        [1, 4]

    Before the call, axis=0 refers to the length-2 dimension and axis=-1 refers
    to the length=4 dimension. After the call, axis=-1 refers to the same
    dimension as before, but axis=0 now refers to a new length=1 dimension. If
    it is desired to compensate for this offset, then instead of passing the
    axes as separate arguments, pass in a single list of the axes indices. This
    list will be modified to offset the axis>=0 appropriately. Ideally, you only
    pass in axes<0, and this does not apply. Doing this in the above example:

        >>> l
        [0, -1, -5]

        >>> x.shape
        (2, 3, 4)

        >>> [x.shape[i] for i in (l[0],l[1])]
        [2, 4]

        >>> x=nps.atleast_dims(x, l)
        >>> x.shape
        (1, 1, 2, 3, 4)

        >>> l
        [2, -1, -5]

        >>> [x.shape[i] for i in (l[0],l[1])]
        [2, 4]

    We passed the axis indices in a list, and this list was modified to reflect
    the new indices: The original axis=0 becomes known as axis=2. Again, if you
    pass in only axis<0, then you don't need to care about this.

    '''

    if any( not isinstance(d, int) for d in dims ):
        if len(dims) == 1 and isinstance(dims[0], list):
            dims = dims[0]
        else:
            raise NumpysaneError("atleast_dims() takes in axes as integers in separate arguments or\n"
                                 "as a single modifiable list")

    if max(dims) >= x.ndim:
        raise NumpysaneError("Axis {} out of bounds because x.ndim={}.\n"
                             "To keep the last dimension anchored, "
                             "only <0 out-of-bounds axes are allowed".format(max(dims), x.ndim))

    need_ndim = -min(d if d<0 else -1 for d in dims)
    if x.ndim >= need_ndim:
        return x
    num_new_axes = need_ndim-x.ndim

    # apply an offset to any axes that need it
    if isinstance(dims, list):
        dims[:] = [ d+num_new_axes if d >= 0 else d for d in dims ]

    return x[ (np.newaxis,)*(num_new_axes) + (_colon,)*x.ndim ]

def mv(x, axis_from, axis_to):
    r'''Moves a given axis to a new position. Similar to numpy.moveaxis().

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.mv( a, -1, 0).shape
        (4, 2, 3)

        >>> nps.mv( a, -1, -5).shape
        (4, 1, 1, 2, 3)

        >>> nps.mv( a, 0, -5).shape
        (2, 1, 1, 3, 4)

    New length-1 dimensions are added at the front, as required, and any axis>=0
    that are passed in refer to the array BEFORE these new dimensions are added.

    '''
    axes = [axis_from, axis_to]
    x = atleast_dims(x, axes)
    return np.moveaxis( x, *axes )

def xchg(x, axis_a, axis_b):
    r'''Exchanges the positions of the two given axes. Similar to numpy.swapaxes()

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.xchg( a, -1, 0).shape
        (4, 3, 2)

        >>> nps.xchg( a, -1, -5).shape
        (4, 1, 2, 3, 1)

        >>> nps.xchg( a, 0, -5).shape
        (2, 1, 1, 3, 4)

    New length-1 dimensions are added at the front, as required, and any axis>=0
    that are passed in refer to the array BEFORE these new dimensions are added.

    '''
    axes = [axis_a, axis_b]
    x = atleast_dims(x, axes)
    return np.swapaxes( x, *axes )

def transpose(x):
    r'''Reverses the order of the last two dimensions.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.transpose(a).shape
        (2, 4, 3)

        >>> nps.transpose( np.arange(3) ).shape
        (3, 1)

    A "matrix" is generally seen as a 2D array that we can transpose by looking
    at the 2 dimensions in the opposite order. Here we treat an n-dimensional
    array as an n-2 dimensional object containing 2D matrices. As usual, the
    last two dimensions contain the matrix.

    New length-1 dimensions are added at the front, as required, meaning that 1D
    input of shape (n,) results in 2D output of shape (1,n).

    '''
    return xchg( atleast_dims(x, -2), -1, -2)

def dummy(x, axis=None):
    r'''Adds a single length-1 dimension at the given position.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.dummy(a, 0).shape
        (1, 2, 3, 4)

        >>> nps.dummy(a, 1).shape
        (2, 1, 3, 4)

        >>> nps.dummy(a, -1).shape
        (2, 3, 4, 1)

        >>> nps.dummy(a, -2).shape
        (2, 3, 1, 4)

        >>> nps.dummy(a, -5).shape
        (1, 1, 2, 3, 4)

    This is similar to numpy.expand_dims(), but handles out-of-bounds dimensions
    better. New length-1 dimensions are added at the front, as required, and any
    axis>=0 that are passed in refer to the array BEFORE these new dimensions
    are added.

    '''
    need_ndim = axis+1 if axis >= 0 else -axis
    if x.ndim >= need_ndim:
        # referring to an axis that already exists. expand_dims() thus works
        return np.expand_dims(x, axis)

    # referring to a non-existing axis. I simply add sufficient new axes, and
    # I'm done
    return atleast_dims(x, axis)

def reorder(x, *dims):
    r'''Reorders the dimensions of an array.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.reorder( a, 0, -1, 1 ).shape
        (2, 4, 3)

        >>> nps.reorder( a, -2 , -1, 0 ).shape
        (3, 4, 2)

        >>> nps.reorder( a, -4 , -2, -5, -1, 0 ).shape
        (1, 3, 1, 4, 2)

    This is very similar to numpy.transpose(), but handles out-of-bounds
    dimensions much better.

    New length-1 dimensions are added at the front, as required, and any axis>=0
    that are passed in refer to the array BEFORE these new dimensions are added.

    '''
    dims = list(dims)
    x = atleast_dims(x, dims)
    return np.transpose(x, dims)

@broadcast_define( (('n',), ('n',)), prototype_output=(), out_kwarg='out' )
def dot(a, b, out=None):
    r'''Non-conjugating dot product of two 1-dimensional n-long vectors.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(3)
        >>> b = a+5
        >>> a
        array([0, 1, 2])

        >>> b
        array([5, 6, 7])

        >>> nps.dot(a,b)
        array(20)

    This is identical to numpysane.inner(). For a conjugating version of this
    function, use nps.vdot().

    '''
    if out is None:
        return np.dot(a,b)

    out.setfield(a.dot(b), out.dtype)
    return out

# nps.inner and nps.dot are equivalent. Set the functionality and update the
# docstring
inner = _clone_function( dot, "inner" )
doc = dot.__doc__
doc = doc.replace("vdot",  "aaa")
doc = doc.replace("dot",   "bbb")
doc = doc.replace("inner", "ccc")
doc = doc.replace("ccc",   "dot")
doc = doc.replace("bbb",   "inner")
doc = doc.replace("aaa",   "vdot")
inner.__doc__ = doc

@broadcast_define( (('n',), ('n',)), prototype_output=(), out_kwarg='out' )
def vdot(a, b, out=None):
    r'''Conjugating dot product of two 1-dimensional n-long vectors.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.array(( 1 + 2j, 3 + 4j, 5 + 6j))
        >>> b = a+5
        >>> a
        array([ 1.+2.j,  3.+4.j,  5.+6.j])

        >>> b
        array([  6.+2.j,   8.+4.j,  10.+6.j])

        >>> nps.vdot(a,b)
        array((136-60j))

        >>> nps.dot(a,b)
        array((24+148j))

    For a non-conjugating version of this function, use nps.dot().

    '''
    if out is None:
        return np.vdot(a,b)

    out.setfield(np.vdot(a,b), out.dtype)
    return out

@broadcast_define( (('n',), ('n',)), prototype_output=('n','n'), out_kwarg='out' )
def outer(a, b, out=None):
    r'''Outer product of two 1-dimensional n-long vectors.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(3)
        >>> b = a+5
        >>> a
        array([0, 1, 2])

        >>> b
        array([5, 6, 7])

        >>> nps.outer(a,b)
        array([[ 0,  0,  0],
               [ 5,  6,  7],
               [10, 12, 14]])
    '''
    if out is None:
        return np.outer(a,b)

    out.setfield(np.outer(a,b), out.dtype)
    return out

@broadcast_define( (('n', 'm'), ('m', 'l')), prototype_output=('n','l'), out_kwarg='out' )
def matmult(a, b, out=None):
    r'''Multiplication of two matrices.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = np.arange(12).reshape(3,4)
        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])

        >>> nps.matmult(a,b)
        array([[20, 23, 26, 29],
               [56, 68, 80, 92]])
    '''
    if out is None:
        return np.dot(a,b)

    out.setfield(a.dot(b), out.dtype)
    return out
