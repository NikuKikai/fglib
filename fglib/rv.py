"""Module for random variables.

This module contains classes for random variables and exceptions.

Classes:
    ParameterException: Exception for invalid parameters.
    RandomVariable: Abstract class for random variables.
    Discrete: Class for discrete random variables.
    Gaussian: Class for Gaussian random variables.

"""

from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod

import numpy as np


class ParameterException(Exception):

    """Exception for invalid parameters."""

    pass


class RandomVariable(ABC):

    """Abstract base class for all random variables."""

    @abstractclassmethod
    def unity(cls, *args):
        """Initialize unit element of the random variable."""

    @abstractproperty
    def dim(self):
        """Dimension of the random variable."""

    @abstractmethod
    def __str__(self):
        """String representation of the random variable."""

    @abstractmethod
    def __add__(self):
        """Addition of two random variables."""

    @abstractmethod
    def __sub__(self):
        """Subtraction of two random variables."""

    @abstractmethod
    def __mul__(self):
        """Multiplication of two random variables."""

    @abstractmethod
    def __iadd__(self):
        """Augmented addition of two random variables."""

    @abstractmethod
    def __isub__(self):
        """Augmented subtraction of two random variables."""

    @abstractmethod
    def __imul__(self):
        """Augmented multiplication of two random variables."""

    @abstractmethod
    def __eq__(self):
        """Compare two random variables for equality."""

    @abstractmethod
    def normalize(self):
        """Normalize the random variable."""

    @abstractmethod
    def marginalize(self):
        """Return marginal of the random variable."""

    @abstractmethod
    def maximize(self):
        """Return maximum of the random variable."""

    @abstractmethod
    def argmax(self):
        """Return dimension of the maximum of the random variable."""

    @abstractmethod
    def log(self):
        """Return natural logarithm of the random variable."""


class Discrete(RandomVariable):

    """Class for discrete random variables.

    A discrete random variable is defined by a single- or multi-dimensional
    probability mass function. In addition, each dimension of the probability
    mass function has to be associated with a variable. The variable is
    represented by a variable node of the comprehensive factor graph.

    """

    def __init__(self, raw_pmf, *args):
        """Initialize a discrete random variable.

        Create a new discrete random variable with the given probability
        mass function over the given variable nodes.

        Args:
            raw_pmf: A Numpy array representing the probability mass function.
                The probability mass function does not need to be normalized.
            *args: Instances of the class VNode representing the variables of
                the probability mass function. The number of the positional
                arguments must match the number of dimensions of the Numpy
                array.

        Raises:
            ParameterException: An error occurred initializing with invalid
                parameters.

        """
        pmf = np.asarray(raw_pmf, dtype=np.float64)

        # Set probability mass function
        self._pmf = pmf

        # Set variable nodes for dimensions
        if np.ndim(pmf) != len(args):
            raise ParameterException('Dimension mismatch.')
        else:
            self._dim = args

    @classmethod
    def unity(cls, *args):
        """Initialize unit element of a discrete random variable.

        Args:
            *args: Instances of the class VNode representing the variables of
                the probability mass function. The number of the positional
                arguments must match the number of dimensions of the Numpy
                array.

        Raises:
            ParameterException: An error occurred initializing with invalid
                parameters.

        """
        n = len(args)
        return cls(np.ones((1,) * n), *args)

    @property
    def pmf(self):
        return self._pmf

    @property
    def dim(self):
        return self._dim

    def __str__(self):
        """Return string representation of the discrete random variable."""
        return np.array2string(self.pmf, separator=',', sign=' ')

    def __add__(self, other):
        """Add other to self and return the result.

        Args:
            other: Summand for the discrete random variable.

        Returns:
            A new discrete random variable representing the summation.

        """
        # Verify dimensions of summand and summand.
        if len(self.dim) < len(other.dim):
            self._expand(other.dim, other.pmf.shape)
        elif len(self.dim) > len(other.dim):
            other._expand(self.dim, self.pmf.shape)

        pmf = self.pmf + other.pmf

        return Discrete(pmf, *self.dim)

    def __sub__(self, other):
        """Subtract other from self and return the result.

        Args:
            other: Subtrahend for the discrete random variable.

        Returns:
            A new discrete random variable representing the subtraction.

        """
        # Verify dimensions of minuend and subtrahend.
        if len(self.dim) < len(other.dim):
            self._expand(other.dim, other.pmf.shape)
        elif len(self.dim) > len(other.dim):
            other._expand(self.dim, self.pmf.shape)

        pmf = self.pmf - other.pmf

        return Discrete(pmf, *self.dim)

    def __mul__(self, other):
        """Multiply other with self and return the result.

        Args:
            other: Multiplier for the discrete random variable.

        Returns:
            A new discrete random variable representing the multiplication.

        """
        # Verify dimensions of multiplicand and multiplier.
        if len(self.dim) < len(other.dim):
            self._expand(other.dim, other.pmf.shape)
        elif len(self.dim) > len(other.dim):
            other._expand(self.dim, self.pmf.shape)

        pmf = self.pmf * other.pmf

        return Discrete(pmf, *self.dim)

    def __iadd__(self, other):
        """Method for augmented addition.

        Args:
            other: Summand for the discrete random variable.

        Returns:
            A new discrete random variable representing the summation.

        """
        return self.__add__(other)

    def __isub__(self, other):
        """Method for augmented subtraction.

        Args:
            other: Subtrahend for the discrete random variable.

        Returns:
            A new discrete random variable representing the subtraction.

        """
        return self.__sub__(other)

    def __imul__(self, other):
        """Method for augmented multiplication.

        Args:
            other: Multiplier for the discrete random variable.

        Returns:
            A new discrete random variable representing the multiplication.

        """
        return self.__mul__(other)

    def __eq__(self, other):
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return np.allclose(self.pmf, other.pmf) \
            and self.dim == other.dim

    def _expand(self, dims, states):
        """Expand dimensions.

        Expand the discrete random variable along the given new dimensions.

        Args:
            dims: List of discrete random variables.

        """
        reps = [1, ] * len(dims)

        # Extract missing dimensions
        diff = [i for i, d in enumerate(dims) if d not in self.dim]

        # Expand missing dimensions
        for d in diff:
            self._pmf = np.expand_dims(self.pmf, axis=d)
            reps[d] = states[d]

        # Repeat missing dimensions
        self._pmf = np.tile(self.pmf, reps)
        self._dim = dims

    def normalize(self):
        """Normalize probability mass function."""
        pmf = self.pmf / np.abs(np.sum(self.pmf))
        return Discrete(pmf, *self.dim)

    def marginalize(self, *dims, normalize=True):
        """Return the marginal for given dimensions.

        The probability mass function of the discrete random variable
        is marginalized along the given dimensions.

        Args:
            *dims: Instances of discrete random variables, which should be
                marginalized out.
            normalize: Boolean flag if probability mass function should be
                normalized after marginalization.

        Returns:
            A new discrete random variable representing the marginal.

        """
        axis = tuple(idx for idx, d in enumerate(self.dim) if d in dims)
        pmf = np.sum(self.pmf, axis)
        if normalize:
            pmf /= np.sum(pmf)

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Discrete(pmf, *new_dims)

    def maximize(self, *dims, normalize=True):
        """Return the maximum for given dimensions.

        The probability mass function of the discrete random variable
        is maximized along the given dimensions.

        Args:
            *dims: Instances of discrete random variables, which should be
                maximized out.
            normalize: Boolean flag if probability mass function should be
                normalized after marginalization.

        Returns:
            A new discrete random variable representing the maximum.

        """
        axis = tuple(idx for idx, d in enumerate(self.dim) if d in dims)
        pmf = np.amax(self.pmf, axis)
        if normalize:
            pmf /= np.sum(pmf)

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Discrete(pmf, *new_dims)

    def argmax(self, dim=None):
        """Return the dimension index of the maximum.

        Args:
            dim: An optional discrete random variable along a marginalization
                should be performed and the maximum is searched over the
                remaining dimensions. In the case of None, the maximum is
                search along all dimensions.

        Returns:
            An integer representing the dimension of the maximum.

        """
        if dim is None:
            return np.unravel_index(self.pmf.argmax(), self.pmf.shape)
        m = self.marginalize(dim)
        return np.argmax(m.pmf)

    def log(self):
        """Natural logarithm of the discrete random variable.

        Returns:
            A new discrete random variable with the natural logarithm of the
            probablitiy mass function.

        """
        return Discrete(np.log(self.pmf), *self.dim)


class Gaussian(RandomVariable):

    """Class for Gaussian random variables.

    A Gaussian random variable is defined by a mean vector and a covariance
    matrix. In addition, each dimension of the mean vector and the covariance
    matrix has to be associated with a variable. The variable is
    represented by a variable node of the comprehensive factor graph.

    """

    def __init__(self, raw_mean, raw_cov, *args):
        """Initialize a Gaussian random variable.

        Create a new Gaussian random variable with the given mean vector and
        the given covariance matrix over the given variable nodes.

        Args:
            raw_mean: A Numpy array representing the mean vector.
            raw_cov: A Numpy array representing the covariance matrix.
            *args: Instances of the class VNode representing the variables of
                the mean vector and covariance matrix, respectively. The number
                of the positional arguments must match the number of dimensions
                of the Numpy arrays.

        Raises:
            ParameterException: An error occurred initializing with invalid
                parameters.

        """
        if raw_mean is not None and raw_cov is not None:
            mean = np.asarray(raw_mean, dtype=np.float64)
            cov = np.asarray(raw_cov, dtype=np.float64)

            # Set mean vector and covariance matrix
            if mean.shape[0] != cov.shape[0]:
                raise ParameterException('Dimension mismatch.')
            else:
                # Precision matrix
                self._W = np.linalg.inv(np.asarray(cov))
                # Precision-mean vector
                self._Wm = np.dot(self._W, np.asarray(mean))

            # Set variable nodes for dimensions
            if cov.shape[0] != len(args):
                raise ParameterException('Dimension mismatch.')
            else:
                self._dim = args

        else:
            self._dim = args

    @classmethod
    def unity(cls, *args):
        """Initialize unit element of a Gaussian random variable.

        Args:
            *args: Instances of the class VNode representing the variables of
                the mean vector and covariance matrix, respectively. The number
                of the positional arguments must match the number of dimensions
                of the Numpy arrays.

        Raises:
            ParameterException: An error occurred initializing with invalid
                parameters.

        """
        n = len(args)
        return cls.inf_form(np.diag(np.ones(n) * 0.001), np.zeros((n, 1)), *args)

    @classmethod
    def inf_form(cls, raw_W, raw_Wm, *args):
        """Initialize a Gaussian random variable using the information form.

        Create a new Gaussian random variable with the given mean vector and
        the given covariance matrix over the given variable nodes.

        Args:
            raw_W: A Numpy array representing the precision matrix.
            raw_Wm: A Numpy array representing the precision-mean vector.
            *args: Instances of the class VNode representing the variables of
                the mean vector and covariance matrix, respectively. The number
                of the positional arguments must match the number of dimensions
                of the Numpy arrays.

        Raises:
            ParameterException: An error occurred initializing with invalid
                parameters.

        """
        g = cls(None, None, *args)
        g._W = np.asarray(raw_W, dtype=np.float64)
        g._Wm = np.asarray(raw_Wm, dtype=np.float64)
        return g

    @property
    def mean(self):
        try:
            return np.dot(np.linalg.inv(self._W), self._Wm)
        except:
            return np.zeros_like(self._Wm)

    @property
    def cov(self):
        try:
            return np.linalg.inv(self._W)
        except:
            return np.ones_like(self._W) * np.inf

    @property
    def dim(self):
        return self._dim

    def __str__(self):
        """Return string representation of the Gaussian random variable."""
        # mean = np.array2string(self.mean, separator=',', sign=' ')
        # cov = np.array2string(self.cov, separator=',', sign=' ')
        # return f"[m={mean}, cov={cov}]"
        wm = np.array2string(self._Wm, separator=',', sign=' ')
        w = np.array2string(self._W, separator=',', sign=' ')
        return f"[Wm={wm}, W={w}]"

    def __add__(self, other):
        """Add other to self and return the result.

        Args:
            other: Summand for the Gaussian random variable.

        Returns:
            A new Gaussian random variable representing the summation.

        """
        return Gaussian(self.mean + other.mean,
                        self.cov + other.cov,
                        *self.dim)

    def __sub__(self, other):
        """Subtract other from self and return the result.

        Args:
            other: Subrahend for the Gaussian random variable.

        Returns:
            A new Gaussian random variable representing the subtraction.

        """
        return Gaussian(self.mean - other.mean,
                        self.cov - other.cov,
                        *self.dim)

    def __mul__(self, other):
        """Multiply other with self and return the result.

        Args:
            other: Multiplier for the Gaussian random variable.

        Returns:
            A new Gaussian random variable representing the multiplication.

        """
        dim = list(self._dim)
        for d in other._dim:
            if d not in dim:
                dim.append(d)
        w_self = np.zeros((len(dim), len(dim)))
        wm_self = np.zeros((len(dim), 1))
        w = np.zeros((len(dim), len(dim)))
        wm = np.zeros((len(dim), 1))

        idxs_self = [dim.index(d) for d in self._dim]
        w_self[np.ix_(idxs_self, idxs_self)] = self._W
        wm_self[np.ix_(idxs_self, [0])] = self._Wm
        idxs_other = [dim.index(d) for d in other._dim]
        w[np.ix_(idxs_other, idxs_other)] = other._W
        wm[np.ix_(idxs_other, [0])] = other._Wm
        w = w + w_self
        wm = wm + wm_self
        return Gaussian.inf_form(w, wm, *dim)

    def __iadd__(self, other):
        """Method for augmented addition.

        Args:
            other: Summand for the Gaussian random variable.

        Returns:
            A new Gaussian random variable representing the summation.

        """
        return self.__add__(other)

    def __isub__(self, other):
        """Method for augmented subtraction.

        Args:
            other: Subtrahend for the Gaussian random variable.

        Returns:
            A new Gaussian random variable representing the subtraction.

        """
        return self.__sub__(other)

    def __imul__(self, other):
        """Method for augmented multiplication.

        Args:
            other: Multiplier for the Gaussian random variable.

        Returns:
            A new Gaussian random variable representing the multiplication.

        """
        return self.__mul__(other)

    def __eq__(self, other):
        """Compare self with other and return the boolean result.

        Two Gaussian random variables are equal only if the mean vectors and
        the covariance matrices are equal and the order of dimensions are
        equal.

        """
        return np.allclose(self._W, other._W) \
            and np.allclose(self._Wm, other._Wm) \
            and self.dim == other.dim

    def normalize(self):
        """Normalize probability density function."""
        return self

    def marginalize(self, *dims, normalize=True):
        """Return the marginal for given dimensions.

        The probability density function of the Gaussian random variable
        is marginalized along the given dimensions.

        Args:
            *dims: Instances of Gaussian random variables, which should be
                marginalized out.
            normalize: Boolean flag if probability mass function should be
                normalized after marginalization.

        Returns:
            A new Gaussian random variable representing the marginal.

        """
        info = self._Wm
        prec = self._W
        axis_a = [idx for idx, d in enumerate(self.dim) if d not in dims]
        axis_b = [idx for idx, d in enumerate(self.dim) if d in dims]
        info_a = info[np.ix_(axis_a, [0])]
        prec_aa = prec[np.ix_(axis_a, axis_a)]
        info_b = info[np.ix_(axis_b, [0])]
        prec_ab = prec[np.ix_(axis_a, axis_b)]
        prec_ba = prec[np.ix_(axis_b, axis_a)]
        prec_bb = prec[np.ix_(axis_b, axis_b)]

        prec_bb_inv = np.linalg.inv(prec_bb)
        info_ = info_a - prec_ab @ prec_bb_inv @ info_b
        prec_ = prec_aa - prec_ab @ prec_bb_inv @ prec_ba

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Gaussian.inf_form(prec_, info_, *new_dims)

    def maximize(self, *dims):
        """Return the maximum for given dimensions.

        The probability density function of the Gaussian random variable
        is maximized along the given dimensions.

        Args:
            *dims: Instances of Gaussian random variables, which should be
                maximized out.

        Returns:
            A new Gaussian random variable representing the maximum.

        """
        axis = tuple(idx for idx, d in enumerate(self.dim) if d not in dims)
        mean = self.mean[np.ix_(axis, [0])]
        cov = self.cov[np.ix_(axis, axis)]

        new_dims = tuple(d for d in self.dim if d not in dims)
        return Gaussian(mean, cov, *new_dims)

    def argmax(self, dim=None):
        """Return the dimension index of the maximum.

        Args:
            dim: An optional Gaussian random variable along a marginalization
                should be performed and the maximum is searched over the
                remaining dimensions. In the case of None, the maximum is
                search along all dimensions.

        Returns:
            An integer representing the dimension of the maximum.

        """
        if dim is None:
            return self.mean
        m = self.marginalize(dim)
        return m.mean

    def log(self):
        """Natural logarithm of the Gaussian random variable.

        Returns:
            A new Gaussian random variable with the natural logarithm of the
            probability density function.

        """
        raise NotImplementedError
