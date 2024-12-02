# Lizard-Pose-Estimation-and-Evaluation

## DLC 
DeepLabCut is an open-source, deep learning-based tool designed for precise, markerless pose estimation of user-defined body parts across various species and behaviors. Developed by Mathis[1] in 2018, it utilizes transfer learning to fine-tune deep neural networks on limited datasets, enabling accurate tracking without the need for physical markers. This approach facilitates the study of complex motor behaviors in naturalistic settings, offering a significant advantage over traditional marker-based systems.

 [1]: https://www.mackenziemathislab.org/deeplabcut

## B-SOiD 

B-SOiD (Behavioral Segmentation of Open-field In Deep Learning) allows users to find behaviors using unsupervised learning, without the need for behavior-annotated data. Specifically, B-SOiD finds clusters in animal behavior using pose estimation data from another tool such as DeepLabCut. B-SOiD begins by extracting pose relationships like distance, speed, and relative angle. Next, it performs a non-linear transformation called UMAP to re-frame data in a lower-dimensional space. Then, HDBSCAN is used to identify clusters, and the clustered features are fed as input to a random forest classifier. In the Python implementation, scikit-learn's RandomForestClassifier is used for this step. Finally, the classifier can used to predict behavior categories in any related data.

![](https://github.com/Human-Augment-Analytics/Lizard-Pose-Estimation-and-Evaluation/blob/main/Behavioral%20Analysis/B-SOiD/Sample%20Gifs/example-side-by-side-shortened.gif)

## VAME

VAME (Video-based Animal Motion Estimation) finds patterns in animal movement with a focus on finding repetitive behaviors. Like B-SOiD, VAME uses pose estimation files from a program like DeepLabCut to identify motifs in animal behavior. VAME first aligns pose estimation data egocentrically and splits the data into fixed-length time windows. Next, it uses a bi-directional recurrent neural network (biRNN) with an encoder-decoder architecture within a Variational Autoencoder (VAE) framework to learn latent representations of the data. After that, both reconstruction and prediction decoders are used to ensure that the latent space captures both reconstruction and prediction. The data is then embedded into the final latent space, and a Hidden Markov Model (HMM) is used to segment the latent space into behavioral motifs.
