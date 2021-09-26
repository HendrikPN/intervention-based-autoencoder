# Interverntion-based Autoencoder

This is an autoencoder that generates a latent representation that is 
disentangled w.r.t. different interventions performed on the generation of its 
observations.

# Requirements
In order to run the neural network code you will require the following:

+ `python 3.9.7`
+ `numpy 1.20.3`
+ `torch 1.8.1`

In order to generate your own data from our example, you will also require the 
following:

+ `scipy 1.7.1`
  
You may run the code with different versions, but these are the versions we have
verified.

# Architecture
**TBA**

# Get Started
In order to train your own intervention-based autoencoder (iAE) you will need to
generate your own data or use the data from our example.

## Train iAE with our data
You can start by running the iAE on the data from our example. Our example can
be found in `/examples/state_tomography.py`. Here, we generate data from 
different types of measurements on a random two-qubit quantum state. 
Measurements are fixed and either act on qubit 1, qubit 2 or both qubits. 
Each datapoint corresponds to a newly created random mixed state.

The data is separated into three different types of intervention:

1. Remove all measurements that affect qubit 2
2. Remove all measurements that affect qubit 1
3. No intervention

To generate this data set for different interventions go to `/examples` and
run in your terminal:

```bash
python state_tomography.py
```

After it has finished, you will see 6 files in the folder `/data`. The following 
three files contain the training data for the different interventions:
1. `tomography_0.pth`
2. `tomography_1.pth`
3. `tomography_2.pth`

Now you are ready to train the iAE on this data! Therefor, go back to the main 
folder. Since we have already hyperparamaterized the code for you, you don't 
need to bother about the paramters in `Config.py` for now. 
Just run the following in your terminal to train your iAE:

```bash
python main.py
```

Now, you should start seeing text that is printed to your console. When it has 
finished, you can review the results in `/results`. There, you will find four 
files:

+ `results_loss_rec.txt`: This is the recreation loss over time for each intervention.
+ `results_loss.txt`: This is the overall loss over time.
+ `results_min.txt`: This is the minimization filter over time.
+ `results_sel.txt`: This is the selection filters of each intervention over time.

The filters will be of particular interest. You can see here which
neurons of the latent representation have been filtered. A value > 0 indicates
that a neuron is ignored by the decoder(s).

If everything went well, you should see in `results_min.txt` something that
looks like this:

```
[-3.0184876918792725, 1.0279388427734375, -3.338914632797241, -4.57394552230835, 1.4305622577667236, -4.4977922439575195, -3.226980209350586, -3.2366509437561035, -4.525698184967041, 1.3006442785263062, -3.021491527557373, -2.9364583492279053, -3.0780858993530273, -4.466502666473389, 0.010970002971589565, -4.533066272735596, -3.0983939170837402, -3.294105291366577, 1.2918269634246826, -4.566442489624023]
```

This is the global "filter value" for each of the 20 starting neurons.
From the last entry, you can see that 15 neurons are still active (indicated by 
a value < 0).

Similarly, you can analyze the results in `results_sel.txt` where entries like
this should populate the file:

```
[Intervention: 0], [5.63159704208374, 4.369762897491455, 5.476593494415283, 3.6281468868255615, 3.538800001144409, -3.7709128856658936, 4.755659103393555, 4.2118096351623535, 5.206002235412598, 4.347437381744385, 5.386545181274414, 5.070257663726807, 3.712785005569458, 4.8540778160095215, 4.232072830200195, -3.8091671466827393, 5.3681535720825195, 5.737484931945801, 4.310754299163818, -3.8430206775665283]
[Intervention: 1], [3.563845634460449, 3.504835367202759, 4.641702651977539, -3.869236946105957, 3.972226858139038, 5.328145980834961, 4.377258777618408, 3.7198026180267334, -3.8097751140594482, 3.416163444519043, 4.593560695648193, 3.4797816276550293, 3.712146759033203, -3.751715660095215, 3.3567497730255127, 4.810200214385986, 3.1704533100128174, 4.839340686798096, 3.2992093563079834, 4.509983539581299]
[Intervention: 2], [-2.6669344902038574, 3.1874544620513916, -2.984987497329712, -3.9072558879852295, 2.919259548187256, -3.827608823776245, -2.886353015899658, -2.8864145278930664, -3.8482534885406494, 3.3910930156707764, -2.6667697429656982, -2.590120792388916, -2.728029727935791, -3.803149461746216, 2.3931884765625, -3.864940643310547, -2.757671356201172, -2.9479329586029053, 3.1411244869232178, -3.896395206451416]
```

These are the "filter values" of each decoder for each of the 20 latent neurons.
Here, we can see that intervention 0 and 1 have only 3 disjoint active neurons 
(corresponding to the 3 variables required for a one-qubit 
[Bloch](https://en.wikipedia.org/wiki/Bloch_sphere) representation). At the same
time, the last decoder, that does not do any intervention, requires all 
information (15 neurons) to reproduce the mixed states.

## Create your own data
**TBA**
