# markov_tpm 

Calculation of the transition probability matrix for a descrete markov process from an unbalanced [panel data](https://en.wikipedia.org/wiki/Panel_data).

Let's assume you have observations of 10 firms and their profits for the last 10 years. 
Let's also assume that some observations are missing. 

```python
>>> import numpy as np
>>> data = list()
>>> for i in range(10):  # generate unbalanced panel
...     a = np.array(list(map(lambda x: float(x), range(10))))
...     a[np.random.randint(0, len(a), size=int(len(a)*0.2))] = np.NaN
...     np.random.shuffle(a)
...     data.append(a)
>>> array = np.array(data).T
>>> print(array)    # randomly shuffled unbalanced panel
[[nan  0.  1.  5.  9.  5. nan  8.  0.  4.]
 [ 5. nan  8.  9.  8.  0.  7.  9.  8.  0.]
 [ 8.  4.  9.  8. nan  3.  3.  1.  3.  1.]
 [ 9.  7.  7. nan  6.  8.  9. nan nan nan]
 [nan  2. nan nan  1.  4.  4.  3. nan  6.]
 [ 4.  5.  4.  7.  3.  9.  8.  5.  9.  7.]
 [ 6.  8.  6.  1.  4. nan  5.  4.  1.  3.]
 [ 3.  3.  0.  4. nan  1.  6.  7.  7.  8.]
 [ 7.  6.  3.  3.  5.  2.  2.  6.  4. nan]
 [ 1.  9. nan  6.  7.  7. nan nan  6.  9.]]
>>> print(array.shape)
(10, 10)
```

The *array* above represents an unbalanced panel data where for the first firm:
 * in the first year we have no data
 * in the second year it's profits were 0
 * in the second year 1
 * etc...
 
Now let's assume we would like to allocate the profits in each year into 5 bins (think of percentiles when number of bins is 100)
and calculate transition probabilities from the observed data.

With the help of this script, we could do the following:

```
>>> import TPM
>>> tpm = TPM.calculate_tpm(array, n_states=5)
>>> tpm
[[0.14285714 0.42857143 0.28571429 0.         0.14285714]
 [0.46153846 0.15384615 0.15384615 0.         0.23076923]
 [0.14285714 0.28571429 0.28571429 0.21428571 0.07142857]
 [0.25       0.25       0.25       0.         0.25      ]
 [0.         0.         0.16666667 0.83333333 0.        ]]
>>> print(tpm.shape)
(5, 5)

``` 
Each entry in the above matrix is a frequency with which firms *jumped* from one state to the other.

**Note**, that *TPM* here stands simply for a class name under which all functions are collected.

## Some Links

If you are interested in markov processes, [here](https://martin-thoma.com/python-markov-chain-packages/)
is a relatively old but good overview of python packages. Additionally, [discreteMarkovChain](https://pypi.org/project/discreteMarkovChain/)
looks like an advanced package.
## Authors

* **Vladimir Korzinov** - *Initial work* - [vvkorz](https://github.com/vvkorz)

## License

MIT License - see the [LICENSE.md](LICENSE) file for details
