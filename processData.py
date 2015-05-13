
# import BayesDB and create client
from bayesdb.client import Client
client = Client()

# specify model and iteration counts
models = 20
iterations = 100
name = 'dfwa'
#filename = 'DistilledFeaturesWithAffect.csv'
filename = 'DFWAshort.csv'

# create a client from the Distilled Features With Affect CSV
client('CREATE BTABLE ' + name + ' FROM ' + filename + ';')

# run analysis
client('INITIALIZE ' + str(models) + ' MODELS FOR ' + name + ';')
client('ANALYZE ' + name + ' FOR ' + str(iterations) + ' ITERATIONS;')


