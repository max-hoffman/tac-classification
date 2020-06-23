# TAC modeling

## Work Overview

### My Progress

I have code and plots for preprocessing and training, but none of my
models could converge. I tried a straight-forward RNN to start off with,
but the nature of the data isn't realy conducive to sequential modeling.
I made progress doing manual and specrogram featurization so I could use
other classification methods, but I did not have time to try a host of
new models and the newly featurized data.

## Code Overview

Replicate my work with the following steps (docker and curl required, if
you get errors building the docker images boost the docker RAM):
```
./scripts/setup.sh
./scripts/train.sh
./scripts/jupyter.sh
```

* `setup.sh` will build the docker images and create some folders
* `train.sh` will preprocess and train a Pytorch model using one of the
  docker images and the pipeline in `Snakefile`
* `jupyter.sh` will expose an endpoint for Jupyter notebook access with
  all of the same dependencies that were used during training

## Methodology

### Comments on Paper

The paper somehow gets 2/3 sober 1/3 inebriated splits for the data,
which is the inverse of what I found. They must have removed most of
the TAC >= .08 data with their filtering. I am not sure if this is valid
or not because I cannot find any source for their preprocessing, but
removing ~half of the data is sort of suspicious.

The paper does not indicate train/test splits by user source, which means
they have a mixture of each user's data in training and testing pools.
The model will overfit on individual user trends in the data,
artificially boosting the published performance.

### Features

One concern with this sort of model is the
user-specific variabiltiy between sober/inebriated state spaces.

For example, I think we can reliably expect a model to identify "person is jumping"
from accelerometer data, but that behavior could be consistent with
either state space (sober/inebriated). Other contextual factors will be
necessary for the model to generalize accurately and quickly without
needing to buildup a timeline of user-specific state spaces (or
requiring user-feedback for real-time training).

Due to the limitations of the problem statement, and the assumption that
we don't have time to customize per individual and store a feature-store
for user-specific state-spaces, I think the model needs as many
contextual features as possible. These could look like:

* hour of day (jumping at 8pm vs 1am)

* is holiday (activity on St. Patty's Day / New Years)

* estimate for place-location of phone signal (at bar vs. at work)

An example of feature-store information that would be helpful (but not
realistic for this demo):

* state-spaces per user (hidden states for recurrent model over time)

* user segmentation vector (i.e. autoencode user historical pattern into
  small vector space relative to all users)

Other context-specific feature that might not be helpful:

* knowing OS type will help generalize at some training data size
  threshold where OS-patterns start to affect minute accelerometer
  differences. Unless we know something extremely specific like "Android has a 100
  millisecod z-axis lag jcompared to iPhone", I think an OS feature is
  a liability for overfitting with a dozen users.

The best way to represent the target value is somewhat ambiguous for
the limited prediction inputs. If we had a more wholistic set of
features for users, I think having a guage of the model's confidence
for outputs would be helpful. If we only ever have 10 data points with a
generic user id, confidence values will mean little:

* Target is the numerical TAC value

* Target is boolean 1/1 for whether target >= .08 or not

* Target is a double between 0 and 1, where we use the blood alcohol
  test's accuracy to convert a raw TAC number to a probability.

Model choice and use-case will affect which of these is best from an
accuracy and usability standpoint. I will stick with boolean values for
now given the sparse nature of the TAC reading data, and switch to
numerical if my model has trouble generating helpful
gradients without a finer-grain loss function. In a production system
with more data I would use either original TAC or probability outputs,
but I am worried about overfitting with so few users to train with.

### Preprocessing

Considerations:

* There is a 10-fold difference between the phones with the least/most
  data. This means we should probably downsample the larger pools.

* There is only TAC data for a subset of the timestamps (ex phone CC6740
  has 2,374,695 accelerometer time-points, but 55 TAC readings). I
  have to extrapolate the target for intermediate timestamps.

* 70% of the windows given are inebriated (24386 / 35139).

