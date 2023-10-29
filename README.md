## icecube-neutrinos-in-deep-ice
## score at 3rd position is achieved.
![icecube-score](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/2a699f3c-3296-4b97-9440-bbbab48ede03)

### Start 
-----
For better understanding of project, read the files in the following order:
1. icecube-eda.ipynb 
2. all_in_one.ipynb
3. icecub-submission-2023-02-01.ipynb.ipynb

### Model
-----
Results present in ice_transparency.txt is obtained from "table of results" present in https://arxiv.org/pdf/1301.5361.pdf.

Three primary components that speedup the process are: 
1. caching the selected data chunks to minimize the substantial cost of data loading and preprocessing, 
2. employing chunk-based random sampling to effectively utilize caching, and
3. Implementing length-matched data sampling (sa​mpling batches with approximately equal lengths) to reduce the computational overhead associated with padding tokens when truncating the batch at the longest sequence.

For events exceeding the maximum considered sequence length, ​p​erformed the following selection process: Initially, ​select randomly detections from the auxiliary=false subset, and if this subset proved insufficient to create a sequence of the required length, then ​sample randomly ​from noisy auxiliary=true detections.
<code>
#- idx, L, self.L => 6, 316, 192
#- .randperm(n) => Returns a random permutation of integers from 0 to n - 1.
ids = torch.randperm(L).numpy()
#- ids[:5] => [ 58 154  81 111 158]
auxiliary_n = np.where(~auxiliary)[0] # return indices where ~auxiliary is True.
auxiliary_p = np.where(auxiliary)[0]    
#- len(ids), len(auxiliary_n), len(auxiliary_p) => 316, 231, 85
#- min(self.L, len(auxiliary_n)) => 192
ids_n = ids[auxiliary_n][: min(self.L, len(auxiliary_n))]
#- min(self.L - len(ids_n), len(auxiliary_p)) => 0
ids_p = ids[auxiliary_p][: min(self.L - len(ids_n), len(auxiliary_p))]
ids = np.concatenate([ids_n, ids_p])
</code>

L denotes the cache_size. 

Groupby "event_id" and aggregate others (i.e., sensor_id, time, charge, auxiliary) to list from "batch_x.parquet" file. This grouped event_id is a single batch to our model. The sensor_id's and other lists are padded/truncated to attain "cache_size". The "mask" tensor has True values with length equal to the length of the actual sensor_id's list (i.e., without pad). Further input L0 also denotes this length value. 

The input pos ([BS,192,3]) denotes x,y and z values (along axis=2) for 192 sensors from "sensor_geometry.csv". "sensor_geometry.csv" have x, y, and z positions for each of the 5160 IceCube sensors. Here the vector (x,y,z) is normalized. 

High Quantum Efficiency is in the lower 50 DOMs. So, assigned sensors["qe"]=1 corresponding rows to these DOMs.

The model is based on building blocks of [BEIT V2](https://arxiv.org/pdf/2208.06366.pdf). The transformer here, considers each event as a sequence of detections. It is crucial to process the provided continuous input (such as time, charge and pos) into a form suitable for transformers. So, "fourier encoding representation" is used, this often used to describe the position in the sequence in language models. This method can be viewed as a soft digitization of the continuous input signal into a set of codes determined by Fourier frequencies. This procedure is applied for all continuous input variables.

Further, the normalized input variables are multiplied by 1024-4096 to have a sufficient resolution after digitizing. For example, after multiplying the normalized time by 4096, the temporal resolution becomes 7.3 ns, and the model may understand smaller time variations because of the continuous nature of the Fourier encoding. This multiplication is critical, and even a change of the coefficient from 128 to 4096 gives 20 bps boost. bps is a measure of data transfer or processing speed. 
<code>
#- time[0][:5] => tensor([-0.1357, -0.1295, -0.1169, -0.1112, -0.0649], device='cuda:0')
#- 4096 * time[0][:5] => tensor([-555.9637, -530.4320, -478.9589, -455.3387, -265.6939], device='cuda:0')        
</code>

Temporal resolution refers to the level of detail or granularity in time-related data, indicating how finely time can be represented or measured. It essentially tells you how small a change in time can be detected or expressed in a given dataset or system.

In special relativity, the spacetime interval between two events in spacetime is defined as:
ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2
Where:
ds is the spacetime interval.
c is the speed of light in a vacuum (a constant).
dt is the time interval between the two events.
dx, dy, and dz are the spatial intervals in the x, y, and z dimensions, respectively. In this equation, the spacetime interval is an invariant quantity, meaning that it is the same for all observers, regardless of their relative velocities. 

For particles moving with a speed close to the speed of light, ​"spacetime interval" is close to zero. What is more useful is that all particles and photons produced in the reactions caused by a single neutrino should have ds^2 close to zero too (it is not fully precise because the refractive index of ice is about 1.3, and the speed of light in it is lower than c). Therefore, by computing ds^2 between all pairs of detections in the given event, it is relatively straightforward to distinguish between detections originating from the same neutrino and those that are merely noise. This criterion can be naturally introduced to a transformer as a relative bias​ ("rel_pos_bias" in model diagram)​ ("class Rel_ds" in models.py). Therefore, during the construction of the attention matrix, the transformer automatically groups detections based on the source event effectively filtering out noise. 

There are 16 transformer blocks in total: 4 blocks with rel_pos_bias + 12 blocks with cls_token. Hyperparameter n_rel decides the number of blocks to whom the rel_pos_bias would be appended.

"Linear(384,1).weight" (in model diagram) acts as a cls_token. The use cls token enables the gradual collection of information about the track direction throughout the entire model.

Trained for the first 2-3 epochs with von Mises-Fisher Loss, and the remaining epochs are performed with using the "competition metric as the objective function" + "0.05 von Mises-Fisher Loss".
![model](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/4776c4d6-42e0-4a92-bbb3-97aa54399b64)

![block_rel_type_1](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/d9e858ed-4d56-4e15-95ab-9bf800c7b5f4)

![block_rel_type_2](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/06675cd0-1880-429d-bf15-99e55b85b24a)

The output [BS,3] is transformed to [azimuth,zenith] form as:
<code>
#- pred.shape => torch.Size([BS, 3])
#- .normalize(input, p=2.0, dim=-1, ...) => Performs Lp normalization of inputs over specified dimension.
pred = F.normalize(pred.float(), dim=-1)   #- converting to unit vectors.
zen = torch.acos(pred[:, 2].clip(-1, 1))
f = F.normalize(pred[:, :2], dim=-1)
az = torch.asin(f[:, 0].clip(-1, 1))
az = torch.where(f[:, 1] > 0, az, math.pi - az)
az = torch.where(az > 0, az, az + 2.0 * math.pi)
</code>

![prediction](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/37bf4d92-6f36-48fc-9a4d-db280f6b0e9a)

#### von Mises-Fisher
In directional statistics, the von Mises–Fisher, is a probability distribution on the (m-1)-sphere in R<sup>m</sup>. If m=2 the distribution reduces to the von Mises distribution on the circle.
The probability density function of the von Mises–Fisher distribution for the random m-dimensional unit vector e(w) is given by:
![von-mises](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/593bf23c-d01a-4f23-8861-7e43f5c95ed1)

The parameters µ and κ are called the mean direction and concentration parameter, respectively. The greater the value of κ, the higher the concentration of the distribution around the mean direction µ. The distribution is “unimodal for κ>0”, “point distribution for κ>0” and “is uniform on the sphere for κ=0”. 
A natural choice for estimating directional distributions is von Mises-Fisher (vMF) distribution defined over a hypersphere of unit norm. That is, a vector close to the mean direction will have high probability. 
The output of the model at each step is a unit vector e_cap of dimension m (=3 in our case). We use κ = ||e_cap||. Thus the density function becomes:
![von-mises-loss](https://github.com/bishnarender/icecube-neutrinos-in-deep-ice/assets/49610834/09be554f-243e-49d2-9df0-62e8d94b8f7f)

Here in our case, kappa is estimated from L2-norm of x,y, and z values. Dot product is computed between prediction ([BS,3]) and target ([BS,3]). The target values (i.e., x,y and z) are obtained from [azimuth,zenith] from train_meta_x.parquet as:
<code>
x = cos(azimuth) * sin(zenith)
y = sin(azimuth) * sin(zenith)
z = cos(zenith)
</code>
[Loss Reference](https://arxiv.org/pdf/1812.04616.pdf)

#### Neutrinos
Neutrinos are subatomic particles with no electric charge, very little mass, and 1/2 unit of spin. 
Neutrinos are often called "ghost particles" because they barely interact with anything else. They are difficult to detect because they have minimal interaction with matter. To detect neutrinos, very large and sensitive detectors are required. Every time atomic nuclei come together (like in the sun) or break apart (like in a nuclear reactor), they produce neutrinos. Even a banana emits neutrinos—they come from the natural radioactivity of the potassium in the fruit.

The "IceCube Neutrino Observatory" is the first detector of its kind, which consists of a cubic kilometer of ice and is designed to search for nearly massless neutrinos. The rare interaction of neutrinos with the matter may produce high-energy particles emitting "Cherenkov radiation", which is recorded with photo-detectors immersed in ice. The objective of this challenge was a reconstruction of the neutrino trace direction based on the photon detections.

#### Cyclical Learning Rates
CLR method allows learning rate to continuously oscillate between reasonable minimum and maximum bounds. Each step has a size (called stepsize), which is the number of iterations (e.g. 1k, 5k, etc) for which learning rate increases/decreases. Two steps form a cycle. Concretely, a CLR cycle with stepsize of 5,000 will consist of 5,000 + 5,000 = 10,000 total iterations. 

Cyclical Learning Rates are effective because they can successfully negotiate saddle points, which typically have small gradients (flat surfaces) and can slow down training when learning rate is small. The best way to overcome such obstacles is to speed up and to move fast until a curved surface is found. The increasing learning rate of CLRs does just that, efficiently.

Super-convergence and 1cycle policy. Super-convergence uses the CLR method, but with just one cycle—which contains two learning rate steps, one increasing and one decreasing—and a large maximum learning rate bound. The cycle’s size must be smaller than the total number of iterations/epochs. After the cycle is complete, the learning rate should decrease even further for the remaining iterations/epochs, several orders of magnitude less than its initial value.

Concretely, in super-convergence, learning rate starts at a low value, increases to a very large value and then decreases to a value much lower than its initial one. A large learning rate acts as a regularisation method. Hence, when using the 1cycle policy, other regularisation methods (batch size, momentum, weight decay, etc) must be reduced.

[Reference](https://iconof.com/1cycle-learning-rate-policy/)











