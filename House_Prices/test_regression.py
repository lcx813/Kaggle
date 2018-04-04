
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics
from datetime import datetime
import scipy.interpolate
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory


# In[2]:

import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics

def acq_max(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'L-BFGS-B' method.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.


    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(100, bounds.shape[0]))

    for x_try in x_tries:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) + BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(BColours.GREEN, BColours.ENDC,
                                                    x[index],
                                                    self.sizes[index] + 2,
                                                    min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass


def matern52(theta, d):
    """
    Matern 5/2 correlation model.::
    
        theta, d --> r(theta, d) = (1+sqrt(5)*r + 5/3*r^2)*exp(-sqrt(5)*r)
        
                               n
            where r = sqrt(   sum  (d_i)^2 / (theta_i)^2 )
                             i = 1
                             
    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the 
        autocorrelation parameter(s).
        
    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.
        
    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation modle.
    """

    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    
    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1
        
    if theta.size == 1:
        r = np.sqrt(np.sum(d ** 2, axis=1)) / theta[0]
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        r = np.sqrt(np.sum(d ** 2 / theta.reshape(1,n_features) ** 2 , axis=1))
        
    return (1 + np.sqrt(5)*r + 5/3.*r ** 2) * np.exp(-np.sqrt(5)*r)
        


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(" Time taken: %i minutes and %s seconds." % (tmin, round(tsec,2)))


class BayesianOptimization(object):

    def __init__(self, f, pbounds, verbose=1):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the original dictionary
        self.pbounds = pbounds

        # Get the name of the parameters
        self.keys = list(pbounds.keys())

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)

        # Some function to be optimized
        self.f = f

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None

        # Counter of iterations
        self.i = 0

        # Since scipy 0.16 passing lower and upper bound to theta seems to be
        # broken. However, there is a lot of development going on around GP
        # is scikit-learn. So I'll pick the easy route here and simple specify
        # only theta0.
        self.gp = GaussianProcess(corr=matern52,
                                  theta0=np.random.uniform(0.001, 0.05, self.dim),
                                  thetaL=1e-5 * np.ones(self.dim),
                                  thetaU=1e0 * np.ones(self.dim),
                                  random_start=30)

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        self.init_points += list(map(list, zip(*l)))

        # Create empty list to store the new values of the function
        y_init = []

        # Evaluate target function at all initialization
        # points (random + explore)
        for x in self.init_points:

            y_init.append(self.f(**dict(zip(self.keys, x))))

            if self.verbose:
                self.plog.print_step(x, y_init[-1])

        # Append any other points passed by the self.initialize method (these
        # also have a corresponding target value passed by the user).
        self.init_points += self.x_init

        # Append the target value of self.initialize method.
        y_init += self.y_init

        # Turn it into np array and store.
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        # Updates the flag
        self.initialized = True

    def explore(self, points_dict):
        """
        Method to explore user defined points

        :param points_dict:
        :return:
        """

        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')

        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        self.init_points = list(map(list, zip(*all_points)))

    def initialize(self, points_dict):
        """
        Method to introduce point for which the target function
        value is known

        :param points_dict:
        :return:
        """

        for target in points_dict:

            self.y_init.append(target)

            all_points = []
            for key in self.keys:
                all_points.append(points_dict[target][key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """

        # Update the internal object stored dict
        self.pbounds.update(new_bounds)

        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.pbounds.keys()):

            # Reset all entries, even if the same.
            self.bounds[row] = self.pbounds[key]

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):

                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])

                pwarning = True

            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))

            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)

            # Print stuff
            if self.verbose:
                self.plog.print_step(self.X[-1], self.Y[-1], warning=pwarning)

            # Keep track of total number of iterations
            self.i += 1

            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        
        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max - xi)/np.sqrt(var)
        return (mean - y_max - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max - xi)/np.sqrt(var)
        return norm.cdf(z)

def XGbcv( max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):

    global RMSEbest
    global ITERbest

    paramt = {
              'booster' : 'gbtree',
              'max_depth' : max_depth.astype(int),
              'gamma' : gamma,
              'eta' : 0.01,
              'objective': 'reg:linear',
              'nthread' : 8,
              'silent' : True,
              'eval_metric': 'rmse',
              'subsample' : subsample,
              'colsample_bytree' : colsample_bytree,
              'min_child_weight' : min_child_weight,
              'max_delta_step' : max_delta_step.astype(int),
              'seed' : 1001
              }

    folds = 5

    xgbr = xgb.cv(
           paramt,
           dtrain,
           num_boost_round = 100000,
#           stratified = True,
           nfold = folds,
           verbose_eval = False,
           early_stopping_rounds = 50,
           metrics = "rmse",
           show_stdv = True
          )

    cv_score = xgbr['test-rmse-mean'].iloc[-1]
    if ( cv_score < RMSEbest ):
        RMSEbest = cv_score
        ITERbest = len(xgbr)

    return (-1.0 * cv_score)


url="https://raw.githubusercontent.com/lcx813/data/master/house_price_train.csv"
train=pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')),na_values=['NA','?'])
url="https://raw.githubusercontent.com/lcx813/data/master/house_price_test.csv"
test=pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')),na_values=['NA','?'])


# In[3]:

train.head(5)
train.shape


# In[4]:

test.head(5)
test.shape


# In[5]:

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[6]:



fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()



# In[7]:

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.shape

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[8]:

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
print(train['SalePrice'])


# In[9]:

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

print(train['SalePrice'])


# In[10]:

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
print(y_train)

print(train['SalePrice'])


# In[11]:

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# In[12]:

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[13]:

#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[14]:

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[15]:

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[16]:

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[17]:

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[18]:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[19]:

skewness = skewness[abs(skewness.Skew)>0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# In[20]:

all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[21]:

train = all_data[:ntrain]
test = all_data[ntrain:]


# In[22]:

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[23]:

numStages = 4
ypred = np.zeros((train.shape[0], numStages))


# In[24]:

ypred.shape


# In[25]:

train.shape


# In[26]:

test.shape


# In[27]:

y_train.shape


# In[28]:

y_train


# In[29]:

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)



# In[30]:

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[31]:

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[32]:

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[33]:

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[34]:
# max_depth=5 gamma=0.000010 min_child_weight=2.953148 max_delta_step=2 subsample=0.664624 colsample_bytree=0.231739

model_xgb = xgb.XGBRegressor(colsample_bytree=0.231739, gamma=0.000010,max_delta_step=2, 
                             learning_rate=0.01, max_depth=5, 
                             min_child_weight=2.953148, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.664624, silent=1,
                             random_state =7, nthread = -1)


# In[35]:

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)






def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))




import pickle
import math        

"""
#########################################################
X_train = train
y = y_train
X_test = test


folds = 5
RMSEbest = 10.
ITERbest = 0
#ids = X_test['Id']

dtrain = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test)

print("\n Train Set Matrix Dimensions: %d x %d" % (X_train.shape[0], X_train.shape[1]))
print("\n Test Set Matrix Dimensions: %d x %d\n" % (X_test.shape[0], X_test.shape[1]))

start_time = timer(None)
print("# Global Optimization Search for XGboost Parameters")
print("\n Please note that negative RMSE values will be shown below. This is because")
print(" RMSE needs to be minimized, while Bayes Optimizer always maximizes the function.\n")

XGbBO = BayesianOptimization(XGbcv, {'max_depth': (3, 10),
                                 'gamma': (0.00001, 1.0),
                                 'min_child_weight': (0, 5),
                                 'max_delta_step': (0, 5),
                                 'subsample': (0.5, 0.9),
                                 'colsample_bytree' :(0.05, 0.4)
                                })

XGbBO.maximize(init_points=10, n_iter=25, acq="ei", xi=0.01)
print("-" * 53)
timer(start_time)

best_RMSE = round((-1.0 * XGbBO.res['max']['max_val']), 6)
max_depth = XGbBO.res['max']['max_params']['max_depth']
gamma = XGbBO.res['max']['max_params']['gamma']
min_child_weight = XGbBO.res['max']['max_params']['min_child_weight']
max_delta_step = XGbBO.res['max']['max_params']['max_delta_step']
subsample = XGbBO.res['max']['max_params']['subsample']
colsample_bytree = XGbBO.res['max']['max_params']['colsample_bytree']

print("\n Best RMSE value: %f" % best_RMSE)
print(" Best XGboost parameters:")
print(" max_depth=%d gamma=%f min_child_weight=%f max_delta_step=%d subsample=%f colsample_bytree=%f" % (int(max_depth), gamma, min_child_weight, int(max_delta_step), subsample, colsample_bytree))



start_time = timer(None)
print("\n# Making Prediction")

paramt = {
          'booster' : 'gbtree',
          'max_depth' : max_depth.astype(int),
          'gamma' : gamma,
          'eta' : 0.01,
          'objective': 'reg:linear',
          'nthread' : 8,
          'silent' : True,
          'eval_metric': 'rmse',
          'subsample' : subsample,
          'colsample_bytree' : colsample_bytree,
          'min_child_weight' : min_child_weight,
          'max_delta_step' : max_delta_step.astype(int),
          'seed' : 1001
          }

xgbr = xgb.train(paramt, dtrain, num_boost_round=int(ITERbest*(1+(1/folds))))

x_true = np.expm1(y)
x_pred = np.expm1(xgbr.predict(dtrain))
# Normalized prediction error clipped to -20% to 20% range
x_diff = np.clip(100 * ( (x_pred - x_true) / x_true ), -20, 20)
plt.figure(1)
plt.title("True vs Predicted Sale Prices")
plt.scatter(x_true, x_pred, c=x_diff)
plt.colorbar()
plt.plot([x_true.min()-5000, x_true.max()+5000], [x_true.min()-5000, x_true.max()+5000], 'k--', lw=1)
plt.xlabel('Sale Price')
plt.ylabel('Predicted Sale Price')
plt.xlim( 0, 800000 )
plt.ylim( 0, 800000 )
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.savefig('./HousePrices-XGb-' + str(folds) + 'fold-train-predictions-01-v2.png')
plt.show(block=False)

y_pred = np.expm1(xgbr.predict(dtest))
result = pd.DataFrame(y_pred, columns=['SalePrice'])
result["Id"] = test_ID
result = result.set_index("Id")
print("\n First 10 Lines of Your Prediction:\n")
print(result.head(10))
now = datetime.now()
sub_file = 'submission_XGb_' + str(best_RMSE) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing Submission File: %s" % sub_file)
result.to_csv(sub_file, index=True, index_label='Id')
timer(start_time)

history_df = pd.DataFrame(XGbBO.res['all']['params'])
history_df2 = pd.DataFrame(XGbBO.res['all']['values'])
history_df = pd.concat((history_df, history_df2), axis=1)
history_df.rename(columns = { 0 : 'RMSE'}, inplace=True)
history_df.index.names = ['Iteration']

x, y, z = history_df['subsample'].values, history_df['colsample_bytree'].values, history_df['RMSE'].values
# Set up a regular grid of interpolation points
xi, yi = np.linspace(0.35, 1.05, 100), np.linspace(0, 0.65, 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function='multiquadric', smooth=0.5)
zi = rbf(xi, yi)

plt.figure(2)
plt.title("Interpolated density distribution of C vs gamma")
plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
       extent=[0.35, 1.05, 0, 0.65], interpolation = 'lanczos')
plt.scatter(x, y, c=z)
plt.colorbar()
plt.xlabel('subsample')
plt.ylabel('colsample_bytree')
plt.savefig('./HousePrices-XGb-' + str(folds) + 'fold-01-v2.png')
plt.show(block=False)
print("\n Optimization Plot Saved:  HousePrices-XGb-%dfold-01-v2.png" % folds)

history_df['RMSE'] = -1.0 * history_df['RMSE']
history_df.to_csv("./HousePrices-XGb-" + str(folds) + "fold-01-v2-grid.csv")
print("\n Grid Search Results Saved:  HousePrices-XGb-%dfold-01-v2-grid.csv\n" % folds)


###########################################################
# Best RMSE value: 0.112454
# Best XGboost parameters:
# max_depth=5 gamma=0.000010 min_child_weight=2.953148 max_delta_step=2 subsample=0.664624 colsample_bytree=0.231739
############################################################
"""
def mm_model_fit():
    T = 2
    for t in range(T):
        print("Start: Iteration %d \n" % (t))
        
        for stage in range(numStages):
            print("Stage %d \n" % (stage))
            target_stage = y_train - ypred[:, np.arange(numStages)!=stage].sum(axis=1)
            print('target_stage')
            print(target_stage)
            
            
            if stage == 0:
                reg = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
                reg.fit(train.values,target_stage)
                #print(lasso.coef_)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
    
                
                if t == 0:
                    print("first step score (RMSE): {}".format(rmsle(y_train,ypred[:,stage])))
                    
                    
            elif stage == 1:
                
                    
                reg = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



                #reg = RandomForestRegressor()
    
                reg.fit(train.values,target_stage)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
            
            elif stage == 2:
                
                    
                reg = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


                
                #reg = RandomForestRegressor()
    
                reg.fit(train.values,target_stage)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
            
            
            else:
                if t == T-1:
                    break
                reg = xgb.XGBRegressor(colsample_bytree=0.231739, gamma=0.000010,max_delta_step=2, 
                             learning_rate=0.01, max_depth=5, 
                             min_child_weight=2.953148, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.664624, silent=1,
                             random_state =7, nthread = -1)

                reg.fit(train.values,target_stage)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
                
    print("final score (RMSE): {}".format(rmsle(y_train,ypred.sum(axis=1))))
                
                
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))




mm_model_fit()     


ypred = np.zeros((test.shape[0], numStages))
for stage in range(numStages):
    
    if stage == 0:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
    elif stage == 1:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
        
    elif stage == 2:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
    
    else:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
    
    
final_ypred = ypred.sum(axis=1)

   
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = np.expm1(final_ypred)
sub.to_csv('submission.csv',index=False)


