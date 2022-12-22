# PU Learning Paper List
This is a repository for PU-learning-based surveys, top ML conference papers and applications in medical imaging analysis.

# Table of Contents
- PU-Learning Paper List
  - Survey
  - ML Conference Paper
  - Application Paper in Medical Image Analysis

# Paper List
## Survey
- A Recent Survey on Instance-Dependent Positive and Unlabeled Learning. [[paper]](https://gcatnjust.github.io/ChenGong/paper/gong_frme22.pdf)
  - Chen Gong, Muhammad Imran Zulfiqar, Chuang Zhang, Shahid Mahmood, Jian Yang. Fundamental Research 2022.
  - Keywords: Instance-dependent positive and unlabeled learning, Weakly supervised learning, Label noise learning, Cost-sensitive learning.
  -  <details><summary>Digest</summary> In this survey, we first present the preliminary knowledge of PU learning, and then review the representative instance-dependent PU learning settings and methods. After that, we thoroughly compare them with typical PU learning methods on various benchmark datasets and analyze their performances. Finally, we discuss the potential directions for future research.


- Learning from positive and unlabeled data: a survey. [[paper]](https://link.springer.com/content/pdf/10.1007/s10994-020-05877-5.pdf?pdf=button)
  - Jessa Bekker, Jesse Davis. Machine Learning 2020.
  - Keywords: Classification, Weakly supervised learning, PU learning.
  - <details><summary>Digest</summary> This article provides a survey of the current state of the art in PU learning. It proposes seven key research questions that commonly arise in this field and provides a broad overview of how the field has tried to address them.

- Positive And Unlabeled Learning Algorithms And Applications: A Survey.  [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8900698) 
  - Kristen Jaskie, Andreas Spanias. IISA 2019.
  - Keywords: PU learning, positive unlabeled learning, machine learning, artificial intelligence, classification.
  - <details><summary>Digest</summary> In this paper, we explore several applications for PU learning including examples in biological/medical, business, security, and signal processing. We then survey the literature for new and existing solutions to the PU learning problem.

  


## ML Conference/Journal Paper
- Federated Learning with Positive and Unlabeled Data.[[paper]](https://proceedings.mlr.press/v162/lin22b/lin22b.pdf)
  - Xinyang Lin, Hanting Chen, Yixing Xu, Chao Xu, Xiaolin Gui, Yiping Deng, Yunhe Wang. ICML 2022.
  - Keywords: Federated learning with Positive and Unlabeled data (FedPU).
  -  <details><summary>Digest</summary> We propose a novel framework, namely Federated learning with Positive and Unlabeled data (FedPU), to minimize the expected risk of multiple negative classes by leveraging the labeled data in other clients. We theoretically analyze the generalization bound of the proposed FedPU.

- Rethinking Class-Prior Estimation for Positive-Unlabeled Learning. [[paper]](https://openreview.net/pdf?id=aYAA-XHKyk)
  - Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Gang Niu, Masashi Sugiyama, Dacheng Tao. ICLR 2022.
  - Keywords: Class-prior estimation, positive-unlabeled learning.
  - <details><summary>Digest</summary>  In this paper, we rethink CPE for PU learning—can we remove the assumption to make CPE always valid? We show an affirmative answer by proposing Regrouping CPE (ReCPE) that builds an auxiliary probability distribution such that the support of the positive data distribution is never contained in the support of the negative data distribution. ReCPE can work with any CPE method by treating it as the base method. 
  
- Positive Unlabeled Learning by Semi-Supervised Learning [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897738)
  - Zhuowei Wang, Jing Jiang, Guodong Long. IEEE ICIP 2022.
  - Keywords: Image Classification, Positive-Unlabeled Learning, Semi-Supervised Learning.
  - <details><summary>Digest</summary> formance degradation problem. To this end, we propose a novel SSL-based framework to tackle PU learning. Firstly, we introduce the dynamic increasing sampling strategy to progressively select both negative and positive samples from U data. Secondly, we adopt MixMatch to take full advantage of the unchosen samples in U data. Finally, we propose the Co-learning strategy that iteratively trains two independent networks with the selected samples to avoid the confirmation bias. 

- Positive Unlabeled Learning with a Sequential Selection Bias. [[paper]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611977172.3)
  - Walter Gerych, Tom Hartvigsen, Luke Buquicchio, Abdulaziz Alajaji, Kavin Chandrasekaran, Hamid Mansoor, Elke Rundensteiner, and Emmanuel Agu. SDM 2022.
  - Keywords: sequential bias, DeepSPU.
  - <details><summary>Digest</summary> In this work, we propose a novel solution to tackling this open sequential bias problem, called DeepSPU. DeepSPU recovers missing labels by constructing a model of the sequentially biased labeling process itself. This labeling model is then learned jointly with the prediction model that infers the missing labels in an iterative training process. Further, we regulate this training using a theoretically-justified cost functions that prevent our model from converging to incorrect but low-cost solution.
  
- Graph-based PU learning for binary and multiclass classification without class prior.  [[paper]](https://link.springer.com/content/pdf/10.1007/s10115-022-01702-8.pdf?pdf=button)
  - Jaemin Yoo, Junghun Kim, Hoyoung Yoon, Geonsoo Kim, Changwon Jang & U Kang. Knowledge and Information Systems 2022.
  - Keywords: Graph-based PU Learning, Risk Minimization, IterAtive Belief Propagation.
  - <details><summary>Digest</summary> In this work, we propose GRAB (Graph-based Risk minimization with iterAtive Belief propagation), a novel end-to-end approach for graph-based PU learning that requires no class prior. GRAB runs marginalization and update steps iteratively. The marginalization step models the given graph as a Markov network and estimates the marginals of latent variables. The update step trains the binary classifier by utilizing the computed marginals in the objective function. We then generalize GRAB to multi-positive unlabeled (MPU) learning, where multiple positive classes exist in a dataset. 
  
- Who Is Your Right Mixup Partner in Positive and Unlabeled Learning. [[paper]](https://openreview.net/pdf?id=NH29920YEmj)
  - Changchun Li, Ximing Li, Lei Feng, Jihong Ouyang. ICLR 2022.
  - Keywords: Positive and Unlabeled Learning, Mixup, Heuristic.
  - <details><summary>Digest</summary> In this paper, we propose a novel PU learning method, namely Positive and unlabeled learning with Partially Positive Mixup (P3Mix), which simultaneously benefits from data augmentation and supervision correction with a heuristic mixup technique. To be specific, we take inspiration from the directional boundary deviation phenomenon observed in our preliminary experiments, where the learned PU boundary tends to deviate from the fully supervised boundary towards the positive side. For the unlabeled instances with ambiguous predictive results, we select their mixup partners from the positive instances around the learned PU boundary, so as to transform them into augmented instances near to the boundary yet with more precise supervision. Accordingly, those augmented instances may push the learned PU boundary towards the fully supervised boundary, thereby improving the classification performance. 

- Predictive Adversarial Learning from Positive and Unlabeled Data. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16953/16760)
  - Wenpeng Hu, Ran Le, Bing Liu, Feng Ji, Jinwen Ma, Dongyan Zhao and Rui Yan. AAAI 2021.
  - Keywords: Classification and Regression.
  - <details><summary>Digest</summary> This paper proposes a novel PU learning method called Predictive Adversarial Networks (PAN) based on GAN (Generative Adversarial Networks). GAN learns a generator to generate data (e.g., images) to fool a discriminator which tries to determine whether the generated data belong to a (positive) training class. PU learning can be casted as trying to identify (not generate) likely positive instances from the unlabeled set to fool a discriminator that determines whether the identified likely positive instances from the unlabeled set are indeed positive. However, directly applying GAN is problematic because GAN focuses on only the positive data. The resulting PU learning method will have high precision but low recall. We propose a new objective function based on KL-divergence. Evaluation using both image and text data shows that PAN outperforms state-of-the-art PU learning methods and also a direct adaptation of GAN for PU learning. We propose a new objective function based on KL-divergence. 

 - PULNS: Positive-Unlabeled Learning with Effective Negative Sample Selector.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17064/16871)
   - Chuan Luo, Pu Zhao, Chen Chen, Bo Qiao, Chao Du, Hongyu Zhang, Wei Wu, Shaowei Cai, Bing He, Saravanakumar Rajmohan and Qingwei Lin. AAAI 2021.
   - Keywords: Semi-Supervised Learning, Unsupervised & Self-Supervised Learning.
   - <details><summary>Digest</summary>  In this paper, we propose a novel PU learning approach dubbed PULNS, equipped with an effective negative sample selector, which is optimized by reinforcement learning. Our PULNS approach employs an effective negative sample selector as the agent responsible for selecting negative samples from the unlabeled data. While the selected, likely negative samples can be used to improve the classifier, the performance of classifier is also used as the reward to improve the selector through the REINFORCE algorithm. By alternating the updates of the selector and the classifier, the performance of both is improved.
  
- Mixture Proportion Estimation and PU Learning: A Modern Approach. [[paper]](https://proceedings.neurips.cc/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf)
  - Saurabh Garg, Yifan Wu, Alex Smola, Sivaraman Balakrishnan, Zachary C. Lipton. NeurIPS 2021.
  - Keywords:  Mixture Proportion Estimation (MPE), Best Bin Estimation (BBE), Conditional Value Ignoring Risk (CVIR).
  - <details><summary>Digest</summary> In this paper, we propose two simple techniques: Best Bin Estimation (BBE) (for MPE); and Conditional Value Ignoring Risk (CVIR), a simple objective for PU-learning. Both methods dominate previous approaches empirically, and for BBE, we establish formal guarantees that hold whenever we can train a model to cleanly separate out a small subset of positive examples. Our final algorithm (TED)^n, alternates between the two procedures, significantly improving both our mixture proportion estimator and classifier. 
  
- Asymmetric Loss for Positive-Unlabeled Learning. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9428350)
  - Cong Wang, Jian Pu, Zhi Xu, and Junping Zhang. IEEE ICME 2021.
  - Keywords: Positive-unlabeled learning, asymmetric loss, deep neural networks.
  - <details><summary>Digest</summary> For the situation with selection bias on the labeled samples, we propose a heuristic method to automatically choose the hyper-parameter according to the class prior on the training data. Compared with previous approaches, our method only requires a slight modification of the conventional cross-entropy loss and is compatible with various deep neural networks in an end-to-end way. 

 - Positive-Unlabeled Learning from Imbalanced Data [[paper]](https://www.ijcai.org/proceedings/2021/0412.pdf)
   - Guangxin Su, Weitong Chen, Miao Xu. IJCAI 2021.
   -  <details><summary>Digest</summary> In this paper, we explore this problem and propose a general learning objective for PU learning targeting specially at imbalanced data. By this general learning objective, state-of- the-art PU methods based on optimizing a consis- tent risk estimator can be adapted to conquer the imbalance. We theoretically show that in expecta- tion, optimizing our learning objective is equivalent to learning a classifier on the oversampled balanced data with both P and N data available, and further provide an estimation error bound. 

- Positive-Unlabeled Learning from Imbalanced Data [[paper]](https://www.ijcai.org/proceedings/2021/0412.pdf)
  - Guangxin Su, Weitong Chen, Miao Xu. IJCAI 2021.
  -  <details><summary>Digest</summary> In this paper, we explore this problem and propose a general learning objective for PU learning targeting specially at imbalanced data. By this general learning objective, state-of- the-art PU methods based on optimizing a consis- tent risk estimator can be adapted to conquer the imbalance. We theoretically show that in expecta- tion, optimizing our learning objective is equivalent to learning a classifier on the oversampled balanced data with both P and N data available, and further provide an estimation error bound. 

- Object detection as a positive-unlabeled problem [[paper]](https://arxiv.org/pdf/2002.04672.pdf)
  - Y Yang, KJ Liang, L Carin. BMVC 2020.
  - Keywords: Label quality, object detection, pattern recognition. 
  - <details><summary>Digest</summary> Label quality is important for learning modern convolutional object detectors. However, the potentially large number and wide diversity of object instances that can be found in complex image scenes makes constituting complete annotations a challenging task; objects missing annotations can be observed in a variety of popular object detection datasets. These missing annotations can be problematic, as the standard cross-entropy loss employed to train object detection models treats classification as a positive-negative (PN) problem: unlabeled regions are implicitly assumed to be background. As such, any object missing a bounding box results in a confusing learning signal, the effects of which we observe empirically. To remedy this, we propose treating object detection as a positive-unlabeled (PU) problem, which removes the assumption that unlabeled regions must be negative. 
 
- Partial Optimal Transport with Applications on Positive-Unlabeled Learning. [[paper]](https://proceedings.neurips.cc/paper/2020/file/1e6e25d952a0d639b676ee20d0519ee2-Paper.pdf) [[code]](https://github.com/TAMU-VITA/Self-PU)
  - Laetitia Chapel, Mokhtar Z. Alaya and Gilles Gasso. NeurIPS 2020.
  - <details><summary>Digest</summary> we address the partial Wasserstein and Gromov-Wasserstein problems and propose exact algorithms to solve them. We showcase the new formulation in a positive-unlabeled (PU) learning application. To the best of our knowledge, this is the first application of optimal transport in this context and we first highlight that partial Wasserstein-based metrics prove effective in usual PU learning settings. We then demonstrate that partial Gromov-Wasserstein metrics are efficient in scenarii in which the samples from the positive and the unlabeled datasets come from different domains or have different features.
  
 - Temporal Positive-unlabeled Learning for Biomedical Hypothesis Generation via Risk Estimation. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/310614fca8fb8e5491295336298c340f-Abstract.html)
   - Uchenna Akujuobi, Jun Chen, Mohamed Elhoseiny, Michael Spranger, Xiangliang Zhang. NeurIPS 2020.
   - Keywords: Biomedical hypothesis generation, risk estimation, temporal positive-unlabeled learning.
   -  <details><summary>Digest</summary> Most existing methods fail to truly capture the temporal dynamics of scientific term relations and also assume unobserved connections to be irrelevant (i.e., in a positive-negative (PN) learning setting). To break these limits, we formulate this HG problem as future connectivity prediction task on a dynamic attributed graph via positive-unlabeled (PU) learning. Then, the key is to capture the temporal evolution of node pair (term pair) relations from just the positive and unlabeled data. We propose a variational inference model to estimate the positive prior, and incorporate it in the learning of node pair embeddings, which are then used for link prediction. 
  
- A Variational Approach for Learning from Positive and Unlabeled Data. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/aa0d2a804a3510442f2fd40f2100b054-Abstract.html)
  - Hui Chen, Fangqing Liu, Yin Wang, Liyue Zhao, Hao Wu. NeurIPS 2020.
  - Keywords: variational PU (VPU) learning, Bayesian classifier.
  - <details><summary>Digest</summary> In this paper, we introduce a variational principle for PU learning that allows us to quantitatively evaluate the modeling error of the Bayesian classiﬁer directly from given data. This leads to a loss function which can be efﬁciently calculated without involving class prior estimation or any other intermediate estimation problems, and the variational learning method can then be employed to optimize the classiﬁer under general conditions. 
  
 - Learning from Positive and Unlabeled Data with Arbitrary Positive Shift
   - Zayd Hammoudeh, Daniel Lowd. NeurIPS 2020.
   - Keywords: Arbitrary positive shift, recursive risk estimator, PU risk estimation.
   -  <details><summary>Digest</summary> This paper shows that PU learning is possible even with arbitrarily non-representative positive data given unlabeled data from the source and target distributions. Our key insight is that only the negative class's distribution need be fixed. We integrate this into two statistically consistent methods to address arbitrary positive bias - one approach combines negative-unlabeled learning with unlabeled-unlabeled learning while the other uses a novel, recursive risk estimator.
  
- Learning from Multi-Class Positive and Unlabeled Data. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9338280)
  - Senlin Shu, Zhuoyi Lin, Yan Yan, Li Li. IEEE ICDM 2020.
  - Keywords: multi-class PU learning, unbiased risk estimator, alternative risk estimator.
  - <details><summary>Digest</summary> In this paper, we present an unbiased estimator of the original classification risk for multi-class PU learning, and show that the direct empirical risk minimization suffers from the severe overfitting problem because the risk is unbounded below. To address this problem, we propose an alternative risk estimator, and theoretically establish an estimation error bound. We show that the estimation error of its empirical risk minimizer achieves the optimal parametric convergence rate. 
  
- Online Positive and Unlabeled Learning. [[paper]](https://www.researchgate.net/profile/Sujit-Gujar/publication/342796625_FNNC_Achieving_Fairness_through_Neural_Networks/links/5f310916458515b7291205ad/FNNC-Achieving-Fairness-through-Neural-Networks.pdf)
  - Chuang Zhang, Chen Gong, Tengfei Liu, Xun Lu, Weiqiang Wang and Jian Yang. IJCAI 2020.
  - Keywords: gradient based online learning, Online Positive and Unlabeled (OPU).
  - <details><summary>Digest</summary> This paper proposes a novel positive and unlabeled learning algorithm in an online training mode, which trains a classifier solely on the positive and unlabeled data arriving in a sequential order. Specifically, we adopt an unbiased estimate for the loss induced by the arriving positive or unlabeled examples at each time. Then we show that for any coming new single datum, the model can be updated independently and incrementally by gradient based online learning method. Furthermore, we extend our method to tackle the cases when more than one example is received at each time. 
  
- Positive and Unlabeled Learning with Label Disambiguation. [[paper]](https://www.ijcai.org/proceedings/2019/0590.pdf)
  - Chuang Zhang, Dexin Ren, Tongliang Liu, Jian Yang and Chen Gong. IJCAI 2019.
  - Keywords: Positive and Unlabeled (PU) learning, label disambiguation.
  - <details><summary>Digest</summary> This paper proposes a novel algorithm dubbed as “Positive and Unlabeled learning with Label Disambiguation” (PULD). We first regard all the unlabeled examples in PU learning as ambiguously labeled as positive and negative, and then employ the margin-based label disambiguation strategy, which enlarges the margin of classifier response between the most likely label and the less likely one, to find the unique ground-truth label of each unlabeled example. Theoretically, we derive the generalization error bound of the proposed method by analyzing its Rademacher complexity. 
  
 - Classification from Positive, Unlabeled and Biased Negative Data [[paper]](https://proceedings.mlr.press/v97/hsieh19c/hsieh19c.pdf)
   - Yu-Guan Hsieh, Gang Niu, Masashi Sugiyama. ICML 2019.
   - Keywords: Weighting algorithm, biased negative data, positive-unlabeled (PU) learning.
   - <details><summary>Digest</summary> This paper studies a novel classification framework which incorporates such biased N (bN) data in PU learning. We provide a method based on empirical risk minimization to address this PUbN classification problem. Our approach can be regarded as a novel example-weighting algorithm, with the weight of each example computed through a preliminary step that draws inspiration from PU learning. We also derive an estimation error bound for the proposed method. 
  
- Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework [[paper]](https://www.researchgate.net/profile/Chen-Li-93/publication/269040485_Towards_Positive_Unlabeled_Learning_for_Parallel_Data_Mining_A_Random_Forest_Framework/links/54a9f99d0cf257a6360d5b7f/Towards-Positive-Unlabeled-Learning-for-Parallel-Data-Mining-A-Random-Forest-Framework.pdf)
  - Chen Li and Xue-Liang Hua. ADMA 2014.
  - Keywords: PU information gain, PU Gini index, random forest, parallel data mining.
  - <details><summary>Digest</summary> We investigate widely used Positive and Unlabeled (PU) learning algorithms including PU information gain and a newly developed PU Gini index combining with popular parallel computing framework - Random Forest (RF), thereby enabling parallel data mining to learn from only positive and unlabeled samples. The proposed framework, termed PURF (Positive Un- labeled Random Forest), is able to learn from positive and unlabeled instances and achieve comparable classifcation performance with RF trained by fully la- beled data through parallel computing according to experiments on both synthetic and real-world UCI datasets. 
  
  
  
## Application Paper in Medical Image Analysis
  
- Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training. [[paper]](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf)
  - Xuxi Chen, Wuyang Chen, Tianlong Chen, Ye Yuan, Chen Gong, Kewei Chen, Zhangyang Wang. ICML 2020.
  - Application: classifying brain images of Alzheimer’s Disease.
  - <details><summary>Digest</summary> This paper proposed a novel Self-PU learning framework, which seamlessly integrates PU learning and self-training. Self-PU highlights three “self”-oriented building blocks: a self-paced training algorithm that adaptively discovers and augments confident positive/negative examples as the training proceeds; a self-reweighted, instance-aware loss; and a self-distillation scheme that introduces teacher-students learning as an effective regularization for PU learning. 

- Embryo Grading With Unreliable L abels Due to Chromosome Abnormalities by Regularized PU Learning With Ranking. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9606688&tag=1)
  - Masashi Nagaya and Norimichi Ukita. IEEE Transactions on Medical Imaging 2022.
  - Keywords: Deep convolutional networks, positive-unlabeled learning, learning-to-rank, mutual information.
  - Application: human embryo grading with its images.
  - <details><summary>Digest</summary> For alleviating an adverse effect of the unreliable labels, our method employs Positive-Unlabeled (PU) learning so that live birth and non-live birth are labeled as positive and unlabeled, respectively, where unlabeled samples contain both positive and negative samples. In our method, this PU learning on a deep CNN is improved by a learning-to-rank scheme. While the original learning-to- rank scheme is designed for positive-negative learning, it is extended to PU learning. Furthermore, overfitting in this PU learning is alleviated by regularization with mutual information.
  
- Positive-Unlabeled Learning for Cell Detection in Histopathology Images with Incomplete Annotations [[paper]](https://arxiv.org/pdf/2106.15918.pdf)
  - Zipei Zhao, Fengqian Pang, Zhiwen Liu, Chuyang Ye. MICCAI 2021.
  - Keywords: Cell detection, positive-unlabeled learning, incomplete annotation.
  - Application: Cell detection in histopathology images.
  - <details><summary>Digest</summary>  In this work, to address the problem of incomplete annotations, we formulate the training of detection networks as a positive-unlabeled learning problem. Specifically, the classification loss in network training is revised to take into account incomplete annotations, where the terms corresponding to negative samples are approximated with the true positive samples and the other sam- ples of which the labels are unknown.
  
- ShapePU: A New PU Learning Framework Regularized by Global Consistency for Scribble Supervised Cardiac Segmentation[[paper]](https://arxiv.org/pdf/2206.02118.pdf)[[code]](https://github.com/BWGZK/ShapePU)
  - Ke Zhang and Xiahai Zhuang. MICCAI 2022.
  - Keywords: Weakly supervised learning, PU learning, Segmentation.
  - Application: Cardiac segmentation.
  - <details><summary>Digest</summary> We propose a new scribble-guided method for cardiac segmentation, based on the Positive-Unlabeled (PU) learning framework and global consistency regularization, and termed as ShapePU. To leverage unlabeled pixels via PU learning, we first present an Expectation-Maximization (EM) algorithm to estimate the proportion of each class in the unlabeled pixels. Given the estimated ratios, we then introduce the marginal probability maximization to identify the classes of unlabeled pixels. To exploit shape knowledge, we apply cutout operations to training images, and penalize the inconsistent segmentation results. 
