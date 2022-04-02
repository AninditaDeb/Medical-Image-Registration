# Medical-Image-Registration

Deformable registration is a fundamental task in a variety of medical imaging studies and has been a topic of active research for decades. In deformable registration, a dense, non-linear correspondence is established between a pair of images, such as 3D magnetic resonance (MR) brain scans.
Traditional registration methods solve an optimization problem for each volume pair by aligning voxels with similar appearance while enforcing constraints on the registration mapping. Unfortunately, solving a pairwise optimization can be computationally intensive, and therefore slow in practice. 

We trying to implement the image registration process using a recently developed library that contains a function using a convolutional neural network (CNN), that takes two n-D input volumes and outputs mapping of all voxels of one volume to another volume.
Here we are trying to divide our work into four categories

1)Implement  Non-learning-based Medical Image Registration
Given two images (which we call moving and fixed), our goal is to find the deformation between them. 
In learning-based methods, we use a network that takes in two images  m  ("moving") and  f and outputs a dense deformation  ϕ . 
Intuitively, this deformation  ϕ  gives us the correspondances between the images, and tells us how to moving the moving image to match up with the fixed image.
In a supervised setting we would have ground truth deformations  ϕgt ,
and we could use a supervised loss like MSE  =∥ϕ−ϕgt∥
2)Medical Image Registration (Learning-based)
The main idea in unsupervised registration is to use loss inspired by classical registration.
Without supervision, how do we know this deformation is good?
(1) We have to enusre sure that  m∘ϕ  ( m  warped by  ϕ ) is close to  f 
(2) regularize  ϕ  (often meaning make sure it's smooth)

