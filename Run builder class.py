from collections import OrderedDict
from collections import namedtuple
from itertools import product     ##product function computes a Cartesian product given multiple list inputs.


class RunBuilder():
    @staticmethod
    def get_runs(params):      ## Since the get_runs() method is static, we can call it using the class itself. We don’t need an instance of the class.

        Run = namedtuple('Run', params.keys())     ## This line creates a new tuple subclass called Run that has 
                                                    # named fields. This Run class is used to encapsulate the data
                                                    # for each of our runs. The field names of this class are set by 
                                                    # the list of names passed to the constructor. First, we are passing 
                                                    # the class name. Then, we are passing the field names, and in our case, 
                                                    # we are passing the list of keys from our dictionary.

        runs = []                             ## First we create a list called runs. Then, we use the product() 
                                               # function from itertools to create the Cartesian product using the 
                                               # values for each parameter inside our dictionary. This gives us a set
                                               # of ordered pairs that define our runs. We iterate over these adding a 
                                               # run to the runs list for each one.
        for v in product(*params.values()):    ## For each value in the Cartesian product we have an ordered 
                                                # tuples. The Cartesian product gives us every ordered pair so we 
                                                # have all possible order pairs of learning rates and batch sizes. 
                                                # When we pass the tuple to the Run constructor, we use the * operator 
                                                # to tell the constructor to accept the tuple values as arguments opposed 
                                                # to the tuple itself.
            runs.append(Run(*v))

        return runs

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
)

params.keys() ## odict_keys(['lr', 'batch_size'])
params.values() ## odict_values([[0.01, 0.001], [1000, 10000]])

runs = RunBuilder.get_runs(params)
runs ## output   [
         #  Run(lr=0.01, batch_size=1000),
         #  Run(lr=0.01, batch_size=10000),
         #  Run(lr=0.001, batch_size=1000),
         #  Run(lr=0.001, batch_size=10000)
         #       ]        
run = runs[0] ## can be accessed like this
print(run.lr, run.batch_size)  ## attributes can be accessed          

for run in RunBuilder.get_runs(params):
    comment = f'-{run}'
    