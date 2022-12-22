# PU Learning Paper List
This is a repository for PU-learning-based surveys, top ML conference papers and applications in medical imaging analysis.

# Table of Contents
- PU-Learning Paper List
  - Survey
  - ML Conference/Journal Paper
  - Application Paper in Medical Image Analysis

# Paper List
## Survey
- Positive Unlabeled Learning [[book]](https://link.springer.com/content/pdf/10.1007/978-3-031-79178-9.pdf?pdf=button)
  - Kristen Jaskie , Andreas Spanias. 2022.

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
 
- Dist-PU: Positive-Unlabeled Learning From a Label Distribution Perspective.[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Dist-PU_Positive-Unlabeled_Learning_From_a_Label_Distribution_Perspective_CVPR_2022_paper.pdf)
  - Yunrui Zhao, Qianqian Xu, Yangbangyan Jiang, Peisong Wen, Qingming Huang. CVPR 2022.
  - Keywords: Label Distribution, entropy minimization, Mixup regularization.
  - <details><summary>Digest</summary> Noticing that the label distribution of unlabeled data is fixed when the class prior is known, it can be naturally used as supervision for the model. Motivated by this, we propose to pursue the label distribution consistency between predicted and ground-truth label distributions, which is formulated by aligning their expectations. Moreover, we further adopt the entropy minimization and Mixup regularization to avoid the trivial solution of the label distribution consistency on unlabeled data and mitigate the consequent confirmation bias.
  
- Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment.[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Incorporating_Semi-Supervised_and_Positive-Unlabeled_Learning_for_Boosting_Full_Reference_Image_CVPR_2022_paper.pdf)
  - Yue Cao, Zhaolin Wan, Dongwei Ren, Zifei Yan, Wangmeng Zuo. CVPR 2022.
  - Keywords: image quality assessment (IQA), mean opinion score (MOS), Semi-supervised learning (SSL), positive-unlabeled (PU) learning.
  - <details><summary>Digest</summary> In this paper, we suggest to incorporate semi-supervised and positive-unlabeled (PU) learning for exploiting unlabeled data while mitigating the adverse effect of outliers. Particularly, by treating all labeled data as positive samples, PU learning is leveraged to identify negative samples (i.e., outliers) from unlabeled data. Semi-supervised learning (SSL) is further deployed to exploit positive unlabeled data by dynamically generating pseudo-MOS. We adopt a dual-branch network including reference and distortion branches. Furthermore, spatial attention is introduced in the reference branch to concentrate more on the informative regions, and sliced Wasserstein distance is used for robust difference map computation to address the misalignment issues caused by images recovered by GAN models. 
  
- Federated Learning with Positive and Unlabeled Data.[[paper]](https://proceedings.mlr.press/v162/lin22b/lin22b.pdf)
  - Xinyang Lin, Hanting Chen, Yixing Xu, Chao Xu, Xiaolin Gui, Yiping Deng, Yunhe Wang. ICML 2022.
  - Keywords: Federated learning with Positive and Unlabeled data (FedPU).
  -  <details><summary>Digest</summary> We propose a novel framework, namely Federated learning with Positive and Unlabeled data (FedPU), to minimize the expected risk of multiple negative classes by leveraging the labeled data in other clients. We theoretically analyze the generalization bound of the proposed FedPU.

- Rethinking Class-Prior Estimation for Positive-Unlabeled Learning. [[paper]](https://openreview.net/pdf?id=aYAA-XHKyk)
  - Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Gang Niu, Masashi Sugiyama, Dacheng Tao. ICLR 2022.
  - Keywords: Class-prior estimation, positive-unlabeled learning.
  - <details><summary>Digest</summary>  In this paper, we rethink CPE for PU learning—can we remove the assumption to make CPE always valid? We show an affirmative answer by proposing Regrouping CPE (ReCPE) that builds an auxiliary probability distribution such that the support of the positive data distribution is never contained in the support of the negative data distribution. ReCPE can work with any CPE method by treating it as the base method. 
  
 - Positive-Unlabeled Learning with Adversarial Data Augmentation for Knowledge Graph Completion.[[paper]](https://arxiv.org/pdf/2205.00904)[[code]](https://link.zhihu.com/?target=https%3A//github.com/lilv98/PUDA-IJCAI22)
   - Zhenwei Tang, Shichao Pei, Zhao Zhang, Yongchun Zhu, Fuzhen Zhuang, Robert Hoehndorf, Xiangliang Zhang. IJCAI 2022.
   - Keywords: Adversarial Data Augmentation, Knowledge Graph Completion.
   - <details><summary>Digest</summary> We propose positive-unlabeled learning with adversarial data augmentation (PUDA) for KGC. In particular, PUDA tailors positive-unlabeled risk estimator for the KGC task to deal with the false negative issue. Furthermore, to address the data sparsity issue, PUDA achieves a data augmentation strategy by unifying adversarial training and positive-unlabeled learning under the positive-unlabeled minimax game.
  
- Recovering The Propensity Score From Biased Positive Unlabeled Data. [[paper]](https://www.aaai.org/AAAI22Papers/AAAI-12934.GerychW.pdf)
  - Gerych W, Hartvigsen T, Buquicchio L, et al. AAAI 2022.
  - Keywords: biased distribution, propensity score.
  - <details><summary>Digest</summary> In this work, we propose two sets of assumptions under which the propensity score can be uniquely determined: one in which no assumption is made on the functional form of the propensity score (requiring assumptions on the data distribution), and the second which loosens the data assumptions while assuming a functional form for the propensity score. We then propose inference strategies for each case.
  
- Unifying Knowledge Base Completion with PU Learning to Mitigate the Observation Bias.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20332/20091)
  - Jonas Schouterden, Jessa Bekker, Jesse Davis, Hendrik Blockeel. AAAI 2022.
  - Keywords: Data Mining & Knowledge Management (DMKM), Machine Learning (ML).
  - <details><summary>Digest</summary> We make three contributions.: (1) We provide a unifying view that formalizes the relationship between multiple existing confidences measures based on (i) what assumption they make about and (ii) how their accuracy depends on the selection mechanism. (2) We introduce two new confidence measures that can mitigate known biases by using propensity scores that quantify how likely a fact is to be included the KB. (3) We show through theoretical and empirical analysis that taking the bias into account improves the confidence estimates, even when the propensity scores are not known exactly.
  
- Positive Unlabeled Learning by Semi-Supervised Learning [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897738)
  - Zhuowei Wang, Jing Jiang, Guodong Long. IEEE ICIP 2022.
  - Keywords: Image Classification, Positive-Unlabeled Learning, Semi-Supervised Learning.
  - <details><summary>Digest</summary> formance degradation problem. To this end, we propose a novel SSL-based framework to tackle PU learning. Firstly, we introduce the dynamic increasing sampling strategy to progressively select both negative and positive samples from U data. Secondly, we adopt MixMatch to take full advantage of the unchosen samples in U data. Finally, we propose the Co-learning strategy that iteratively trains two independent networks with the selected samples to avoid the confirmation bias. 

- Positive Unlabeled Learning with a Sequential Selection Bias. [[paper]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611977172.3)
  - Walter Gerych, Tom Hartvigsen, Luke Buquicchio, Abdulaziz Alajaji, Kavin Chandrasekaran, Hamid Mansoor, Elke Rundensteiner, and Emmanuel Agu. SDM 2022.
  - Keywords: sequential bias, DeepSPU.
  - <details><summary>Digest</summary> In this work, we propose a novel solution to tackling this open sequential bias problem, called DeepSPU. DeepSPU recovers missing labels by constructing a model of the sequentially biased labeling process itself. This labeling model is then learned jointly with the prediction model that infers the missing labels in an iterative training process. Further, we regulate this training using a theoretically-justified cost functions that prevent our model from converging to incorrect but low-cost solution.
 
- A new method for positive and unlabeled learning with privileged information.[[paper]](https://link.springer.com/content/pdf/10.1007/s10489-021-02528-7.pdf?pdf=button)
  - Bo Liu, Qian Liu & Yanshan Xiao. Applied Intelligence 2022.
  - Keywords: positive and unlabeled learning with privileged information (SPUPIL).
  - <details><summary>Digest</summary> In this paper, we propose a new method, which is based on similarity approach for the problem of positive and unlabeled learning with privileged information (SPUPIL), which consists of two steps. The proposed SPUPIL method first conducts KNN method to generate the similarity weights and then the similarity weights and privileged information are incorporated to the learning model based on Ranking SVM to build a more accurate classifier. We also use the Lagrangian method to transform the original model into its dual problem, and solve it to obtain the classifier.
  
- Graph-based PU learning for binary and multiclass classification without class prior.[[paper]](https://link.springer.com/content/pdf/10.1007/s10115-022-01702-8.pdf?pdf=button)
  - Jaemin Yoo, Junghun Kim, Hoyoung Yoon, Geonsoo Kim, Changwon Jang & U Kang. Knowledge and Information Systems 2022.
  - Keywords: Graph-based PU Learning, Risk Minimization, IterAtive Belief Propagation.
  - <details><summary>Digest</summary> In this work, we propose GRAB (Graph-based Risk minimization with iterAtive Belief propagation), a novel end-to-end approach for graph-based PU learning that requires no class prior. GRAB runs marginalization and update steps iteratively. The marginalization step models the given graph as a Markov network and estimates the marginals of latent variables. The update step trains the binary classifier by utilizing the computed marginals in the objective function. We then generalize GRAB to multi-positive unlabeled (MPU) learning, where multiple positive classes exist in a dataset. 
  
- Who Is Your Right Mixup Partner in Positive and Unlabeled Learning. [[paper]](https://openreview.net/pdf?id=NH29920YEmj)
  - Changchun Li, Ximing Li, Lei Feng, Jihong Ouyang. ICLR 2022.
  - Keywords: Positive and Unlabeled Learning, Mixup, Heuristic.
  - <details><summary>Digest</summary> In this paper, we propose a novel PU learning method, namely Positive and unlabeled learning with Partially Positive Mixup (P3Mix), which simultaneously benefits from data augmentation and supervision correction with a heuristic mixup technique. To be specific, we take inspiration from the directional boundary deviation phenomenon observed in our preliminary experiments, where the learned PU boundary tends to deviate from the fully supervised boundary towards the positive side. For the unlabeled instances with ambiguous predictive results, we select their mixup partners from the positive instances around the learned PU boundary, so as to transform them into augmented instances near to the boundary yet with more precise supervision. Accordingly, those augmented instances may push the learned PU boundary towards the fully supervised boundary, thereby improving the classification performance. 
  
- Positive-Unlabeled Data Purification in the Wild for Object Detection.[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Positive-Unlabeled_Data_Purification_in_the_Wild_for_Object_Detection_CVPR_2021_paper.pdf)
  - Jianyuan Guo, Kai Han, Han Wu, Chao Zhang, Xinghao Chen, Chunjing Xu, Chang Xu, Yunhe Wang. CVPR 2021.
  - Keywords: Data Purification, Object Detection, positive-unlabeled learning.
  - <details><summary>Digest</summary> In this paper, we present a positive-unlabeled learning based scheme to expand training data by purifying valuable images from massive unlabeled ones, where the original training data are viewed as positive data and the unlabeled images in the wild are unlabeled data. To effectively utilized these purified data, we propose a self-distillation algorithm based on hint learning and ground truth bounded knowledge distillation.

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
 
- Positive-Unlabeled Learning from Imbalanced Data [[paper]](https://www.ijcai.org/proceedings/2021/0412.pdf)
   - Guangxin Su, Weitong Chen, Miao Xu. IJCAI 2021.
   - Keywords: risk estimator, classifier learning, oversampled balanced data.
   -  <details><summary>Digest</summary> In this paper, we explore this problem and propose a general learning objective for PU learning targeting specially at imbalanced data. By this general learning objective, state-of- the-art PU methods based on optimizing a consistent risk estimator can be adapted to conquer the imbalance. We theoretically show that in expectation, optimizing our learning objective is equivalent to learning a classifier on the oversampled balanced data with both P and N data available, and further provide an estimation error bound. 

- ARTA: Collection and Classification of Ambiguous Requests and Thoughtful Actions.[[paper]](https://arxiv.org/pdf/2106.07999.pdf) [[code]](https://github.com/ahclab/arta_corpus)
  - Shohei Tanaka, Koichiro Yoshino, Katsuhito Sudoh, Satoshi Nakamura. ACL 2021.
  - Keywords:  dialogue systems, dialogue agent, PU learning.
  - <details><summary>Digest</summary> We collected a corpus and developed a model that classifies ambiguous user requests into corresponding system actions. In order to collect a high-quality corpus, we asked workers to input antecedent user requests whose pre-defined actions could be regarded as thoughtful. Although multiple actions could be identified as thoughtful for a single user request, annotating all combinations of user requests and system actions is impractical. For this reason, we fully annotated only the test data and left the annotation of the training data incomplete. In order to train the classification model on such training data, we applied the positive/unlabeled (PU) learning method, which assumes that only a part of the data is labeled with positive examples. 
  
- Data-Free Knowledge Distillation with Positive-Unlabeled Learning.[[paper]](https://link.springer.com/content/pdf/10.1007/978-3-030-92270-2.pdf?pdf=button)
  - Keywords: Model compression, Data-free knowledge distillation, Positive-unlabeled learning, Attention mechanism.
  - <details><summary>Digest</summary> In this paper, we propose a data-free knowledge distillation method called DFPU, which introduce positive-unlabeled (PU) learning. For training a compact neural network without data, a generator is introduced to generate pseudo data under the supervision of the teacher network. By feeding the generated data into the teacher network and student network, the attention features are extracted for knowledge transfer. The student network is promoted to produce more similar features to the teacher network by PU learning. Without any data, the efficient student network trained by DFPU contains only half parameters and calculations of the teacher network and achieves an accuracy similar to the teacher network.
  
- Asymmetric Loss for Positive-Unlabeled Learning. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9428350)
  - Cong Wang, Jian Pu, Zhi Xu, and Junping Zhang. IEEE ICME 2021.
  - Keywords: Positive-unlabeled learning, asymmetric loss, deep neural networks.
  - <details><summary>Digest</summary> For the situation with selection bias on the labeled samples, we propose a heuristic method to automatically choose the hyper-parameter according to the class prior on the training data. Compared with previous approaches, our method only requires a slight modification of the conventional cross-entropy loss and is compatible with various deep neural networks in an end-to-end way. 

- PUNet: Temporal Action Proposal Generation With Positive Unlabeled Learning Using Key Frame Annotations. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506012)[[code]](https://github.com/NoorZia/punet)
  - Noor Ul Sehr Zia; Osman Semih Kayhan; Jan van Gemert. IEEE ICIP 2021.
  - Keywords: PUNet, Temporal Action Proposal Generation, Key Frame Annotations, Positive Unlabeled Learning.
  - <details><summary>Digest</summary>  To tackle the class imbalance by using only a single frame, we evaluate an extremely simple Positive-Unlabeled algorithm (PU-learning). We demonstrate on THUMOS’14 and ActivityNet that using a single key frame label give good results while being significantly faster to annotate. In addition, we show that our simple method, PUNet, is data-efficient which further reduces the need for expensive annotations. 
  
- Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning.[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8839365)
  - Chen Gong, Hong Shi, Tongliang Liu, Chuang Zhang, Jian Yang and Dacheng Tao. IEEE Transactions on Pattern Analysis and Machine Intelligence 2021.
  - Keywords: Positive and Unlabeled learning, Loss Decomposition and Centroid Estimation (LDCE), Kernelized LDCE (KLDCE).
  - <details><summary>Digest</summary> we propose a novel PU learning algorithm dubbed “Loss Decomposition and Centroid Estimation” (LDCE). By decomposing the loss function of corrupted negative examples into two parts, we show that only the second part is affected by the noisy labels. Thereby, we may estimate the centroid of corrupted negative set via an unbiased way to reduce the adverse impact of such label noise. Furthermore, we propose the “Kernelized LDCE” (KLDCE) by introducing the kernel trick, and show that KLDCE can be easily solved by combining Alternative Convex Search (ACS) and Sequential Minimal Optimization (SMO). Theoretically, we derive the generalization error bound which suggests that the generalization risk of our model converges to the empirical risk with the order of O(1= k + 1= n k+ 1=%/n) (n and k are the amounts of training data and positive data correspondingly). 
  
- Object detection as a positive-unlabeled problem [[paper]](https://arxiv.org/pdf/2002.04672.pdf)
  - Y Yang, KJ Liang, L Carin. BMVC 2020.
  - Keywords: Label quality, object detection, pattern recognition. 
  - <details><summary>Digest</summary> Label quality is important for learning modern convolutional object detectors. However, the potentially large number and wide diversity of object instances that can be found in complex image scenes makes constituting complete annotations a challenging task; objects missing annotations can be observed in a variety of popular object detection datasets. These missing annotations can be problematic, as the standard cross-entropy loss employed to train object detection models treats classification as a positive-negative (PN) problem: unlabeled regions are implicitly assumed to be background. As such, any object missing a bounding box results in a confusing learning signal, the effects of which we observe empirically. To remedy this, we propose treating object detection as a positive-unlabeled (PU) problem, which removes the assumption that unlabeled regions must be negative. 
 
- Partial Optimal Transport with Applications on Positive-Unlabeled Learning. [[paper]](https://proceedings.neurips.cc/paper/2020/file/1e6e25d952a0d639b676ee20d0519ee2-Paper.pdf) [[code]](https://github.com/TAMU-VITA/Self-PU)
  - Laetitia Chapel, Mokhtar Z. Alaya and Gilles Gasso. NeurIPS 2020.
  - Keywords: partial Wasserstein, Gromov-Wasserstein.
  - <details><summary>Digest</summary> we address the partial Wasserstein and Gromov-Wasserstein problems and propose exact algorithms to solve them. We showcase the new formulation in a positive-unlabeled (PU) learning application. To the best of our knowledge, this is the first application of optimal transport in this context and we first highlight that partial Wasserstein-based metrics prove effective in usual PU learning settings. We then demonstrate that partial Gromov-Wasserstein metrics are efficient in scenarii in which the samples from the positive and the unlabeled datasets come from different domains or have different features.
  
 - Temporal Positive-unlabeled Learning for Biomedical Hypothesis Generation via Risk Estimation. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/310614fca8fb8e5491295336298c340f-Abstract.html)
   - Uchenna Akujuobi, Jun Chen, Mohamed Elhoseiny, Michael Spranger, Xiangliang Zhang. NeurIPS 2020.
   - Keywords: Biomedical hypothesis generation, risk estimation, temporal positive-unlabeled learning.
   -  <details><summary>Digest</summary> Most existing methods fail to truly capture the temporal dynamics of scientific term relations and also assume unobserved connections to be irrelevant (i.e., in a positive-negative (PN) learning setting). To break these limits, we formulate this HG problem as future connectivity prediction task on a dynamic attributed graph via positive-unlabeled (PU) learning. Then, the key is to capture the temporal evolution of node pair (term pair) relations from just the positive and unlabeled data. We propose a variational inference model to estimate the positive prior, and incorporate it in the learning of node pair embeddings, which are then used for link prediction. 
  
- A Variational Approach for Learning from Positive and Unlabeled Data. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/aa0d2a804a3510442f2fd40f2100b054-Abstract.html)
  - Hui Chen, Fangqing Liu, Yin Wang, Liyue Zhao, Hao Wu. NeurIPS 2020.
  - Keywords: variational PU (VPU) learning, Bayesian classifier.
  - <details><summary>Digest</summary> In this paper, we introduce a variational principle for PU learning that allows us to quantitatively evaluate the modeling error of the Bayesian classiﬁer directly from given data. This leads to a loss function which can be efﬁciently calculated without involving class prior estimation or any other intermediate estimation problems, and the variational learning method can then be employed to optimize the classiﬁer under general conditions. 
  
 - Learning from Positive and Unlabeled Data with Arbitrary Positive Shift. [[paper]](https://proceedings.neurips.cc/paper/2020/file/98b297950041a42470269d56260243a1-Paper.pdf)
   - Zayd Hammoudeh, Daniel Lowd. NeurIPS 2020.
   - Keywords: Arbitrary positive shift, recursive risk estimator, PU risk estimation.
   -  <details><summary>Digest</summary> This paper shows that PU learning is possible even with arbitrarily non-representative positive data given unlabeled data from the source and target distributions. Our key insight is that only the negative class's distribution need be fixed. We integrate this into two statistically consistent methods to address arbitrary positive bias - one approach combines negative-unlabeled learning with unlabeled-unlabeled learning while the other uses a novel, recursive risk estimator.
  
- Online Positive and Unlabeled Learning. [[paper]](https://www.researchgate.net/profile/Sujit-Gujar/publication/342796625_FNNC_Achieving_Fairness_through_Neural_Networks/links/5f310916458515b7291205ad/FNNC-Achieving-Fairness-through-Neural-Networks.pdf)
  - Chuang Zhang, Chen Gong, Tengfei Liu, Xun Lu, Weiqiang Wang and Jian Yang. IJCAI 2020.
  - Keywords: gradient based online learning, Online Positive and Unlabeled (OPU).
  - <details><summary>Digest</summary> This paper proposes a novel positive and unlabeled learning algorithm in an online training mode, which trains a classifier solely on the positive and unlabeled data arriving in a sequential order. Specifically, we adopt an unbiased estimate for the loss induced by the arriving positive or unlabeled examples at each time. Then we show that for any coming new single datum, the model can be updated independently and incrementally by gradient based online learning method. Furthermore, we extend our method to tackle the cases when more than one example is received at each time. 
  
- Positive Unlabeled Learning with Class-prior Approximation.[[paper]](https://www.ijcai.org/Proceedings/2020/0279.pdf)
  - Shizhen Chang, Bo Du and Liangpei Zhang. IJCAI 2020.
  - Keywords: Positive Unlabeled Learning, Class-prior Approximation, empirical unbiased risk.
  -  <details><summary>Digest</summary> In this paper, we formulate a convex formulation to jointly solve the class-prior unknown problem and train an accurate classifier with no need of any class-prior assumptions or additional negative samples. The class prior is estimated by pursuing the optimal solution of gradient thresholding and the classifier is simultaneously trained by performing empirical unbiased risk.
  
- Class Prior Estimation with Biased Positives and Unlabeled Examples.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5848/5704)
  - Shantanu Jain, Justin Delano, Himanshu Sharma, Predrag Radivojac. AAAI 2020.
  - Keywords: Class Prior Estimation, Biased Data.
  - <details><summary>Digest</summary> In many application domains, however, certain regions in the support of the positive class-conditional distribution are over-represented while others are under-represented in the positive sample. We begin to address this challenge by focusing on the estimation of class priors, quantities central to the estimation of posterior probabilities and the recovery of true classification performance. We start by making a set of assumptions to model the sampling bias. We then extend the identifiability theory of class priors from the unbiased to the biased setting. Finally, we derive an algorithm for estimating the class priors that relies on clustering to decompose the original problem into subproblems of unbiased positive-unlabeled learning.
  
- Fast Nonparametric Estimation of Class Proportions in the Positive-Unlabeled Classification Setting.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6151/6007)
  - Daniel Zeiberg, Shantanu Jain, Predrag Radivojac. AAAI 2020.
  - Keywords: Class Proportions Estimation.
  - <details><summary>Digest</summary> Our algorithm uses a sampling strategy to repeatedly (1) draw an example from the set of positives, (2) record the minimum distance to any of the unlabeled examples, and (3) remove the nearest unlabeled example. We show that the point of sharp increase in the recorded distances corresponds to the desired proportion of positives in the unlabeled set and train a deep neural network to identify that point. Our distance-based algorithm is evaluated on forty datasets and compared to all currently available methods. 
  
- Improving Neural Relation Extraction with Positive and Unlabeled Learning.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6300/6156)
  - Zhengqiu He, Wenliang Chen, Yuyi Wang, Wei Zhang, Guanchun Wang, Min Zhang. AAAI 2020.
  - Keywords: Neural Relation Extraction.
  - <details><summary>Digest</summary> This approach first applies reinforcement learning to decide whether a sentence is positive to a given relation, and then positive and unlabeled bags are constructed. In contrast to most previous studies, which mainly use selected positive instances only, we make full use of unlabeled instances and propose two new representations for positive and unlabeled bags. These two representations are then combined in an appropriate way to make bag-level prediction. 
  
- Social Media Relevance Filtering Using Perplexity-Based Positive-Unlabelled Learning.[[paper]](https://ojs.aaai.org/index.php/ICWSM/article/view/7307/7161)
  - Sunghwan Mac Kim, Stephen Wan, Cécile Paris, Andreas Duenser. AAAI 2020.
  - Keywords: Social Media Relevance Filtering, Perplexity variant of Positive-Unlabelled Learning (PPUL).
  - <details><summary>Digest</summary> In this paper, we introduce our Perplexity variant of Positive-Unlabelled Learning (PPUL) framework as a means to perform social media relevance filtering. We note that this task is particularly well suited to a PU Learning approach. We demonstrate how perplexity can identify candidate examples of the negative class, using language models. To learn such models, we experiment with both statistical methods and a Variational Autoencoder. 
  
- Learning from Multi-Class Positive and Unlabeled Data. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9338280)
  - Senlin Shu, Zhuoyi Lin, Yan Yan, Li Li. IEEE ICDM 2020.
  - Keywords: multi-class PU learning, unbiased risk estimator, alternative risk estimator.
  - <details><summary>Digest</summary> In this paper, we present an unbiased estimator of the original classification risk for multi-class PU learning, and show that the direct empirical risk minimization suffers from the severe overfitting problem because the risk is unbounded below. To address this problem, we propose an alternative risk estimator, and theoretically establish an estimation error bound. We show that the estimation error of its empirical risk minimizer achieves the optimal parametric convergence rate. 
  
- Centroid Estimation With Guaranteed Efficiency: A General Framework for Weakly Supervised Learning. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9294086)
  - Chen Gong; Jian Yang; Jane You; Masashi Sugiyama.  IEEE Transactions on Pattern Analysis and Machine Intelligence 2020.
  - Keywords: Estimation, Supervised learning, Fasteners, Training data, Support vector machines, Semisupervised learning, Safety.
  - <details><summary>Digest</summary> In this paper, we propose a general framework termed centroid estimation with guaranteed efficiency (CEGE) for weakly supervised learning (WSL) with incomplete, inexact, and inaccurate supervision.  The core of our framework is to devise an unbiased and statistically efficient risk estimator that is applicable to various weak supervision. 
  
- Collective loss function for positive and unlabeled learning. [[paper]](https://arxiv.org/pdf/2005.03228.pdf)
  - Chenhao Xie, Qiao Cheng, Jiaqing Liang, Lihan Chen, Yanghua Xiao. CoRR 2020.
  - Keywords: Collectively loss, PU learning, cPU.
  - <details><summary>Digest</summary> In this paper, we propose a Collectively loss function to learn from only Positive and Unlabeled data (cPU). We theoretically elicit the loss function from the setting of PU learning. 

- Counter-examples generation from a positive unlabeled image dataset.[[paper]](https://doi.org/10.1016/j.patcog.2020.107527)
  - Florent Chiaroni, Ghazaleh Khodabandelou, Mohamed-Cherif Rahal, Nicolas Hueber, Frederic Dufaux. Pattern Recognition 2020.
  - Keywords: Counter-examples generation, GAN, positive unlabeled (PU) learning.
  - <details><summary>Digest</summary>  In this context, we propose a two-stage GAN-based model. More specifically, the main contribution is to incorporate a biased PU risk within the standard GAN discriminator loss function. In this manner, the discriminator is constrained to steer the generator to converge towards the unlabeled samples distribution while diverging from the positive samples distribution. Consequently, the proposed model, referred to as D-GAN, exclusively learns the counter-examples distribution without prior knowledge. 
  
- Principled analytic classifier for positive-unlabeled learning via weighted integral probability metric. [[paper]](https://link.springer.com/content/pdf/10.1007/s10994-019-05836-9.pdf?pdf=button)
  - Yongchan Kwon, Wonyoung Kim, Masashi Sugiyama & Myunghee Cho Paik. Machine Learning 2020.
  - Keywords: Principled analytic classifier, positive-unlabeled learning, weighted integral probability metric.
  -  <details><summary>Digest</summary>  In this paper, we propose a computationally efficient and theoretically grounded PU learning algorithm. The proposed PU learning algorithm produces a closed-form classifier when the hypothesis space is a closed ball in reproducing kernel Hilbert space. In addition, we establish upper bounds of the estimation error and the excess risk. The obtained estimation error bound is sharper than existing results and the derived excess risk bound has an explicit form, which vanishes as sample sizes increase. 
  
- Positive and Unlabeled Learning with Label Disambiguation. [[paper]](https://www.ijcai.org/proceedings/2019/0590.pdf)
  - Chuang Zhang, Dexin Ren, Tongliang Liu, Jian Yang and Chen Gong. IJCAI 2019.
  - Keywords: Positive and Unlabeled (PU) learning, label disambiguation.
  - <details><summary>Digest</summary> This paper proposes a novel algorithm dubbed as “Positive and Unlabeled learning with Label Disambiguation” (PULD). We first regard all the unlabeled examples in PU learning as ambiguously labeled as positive and negative, and then employ the margin-based label disambiguation strategy, which enlarges the margin of classifier response between the most likely label and the less likely one, to find the unique ground-truth label of each unlabeled example. Theoretically, we derive the generalization error bound of the proposed method by analyzing its Rademacher complexity. 
  
 - Classification from Positive, Unlabeled and Biased Negative Data [[paper]](https://proceedings.mlr.press/v97/hsieh19c/hsieh19c.pdf)
   - Yu-Guan Hsieh, Gang Niu, Masashi Sugiyama. ICML 2019.
   - Keywords: Weighting algorithm, biased negative data, positive-unlabeled (PU) learning.
   - <details><summary>Digest</summary> This paper studies a novel classification framework which incorporates such biased N (bN) data in PU learning. We provide a method based on empirical risk minimization to address this PUbN classification problem. Our approach can be regarded as a novel example-weighting algorithm, with the weight of each example computed through a preliminary step that draws inspiration from PU learning. We also derive an estimation error bound for the proposed method. 

- Beyond the Selected Completely at Random Assumption for Learning from Positive and Unlabeled Data.[[paper]](https://link.springer.com/content/pdf/10.1007/978-3-030-46147-8.pdf?pdf=button)
  - Jessa Bekker, Pieter Robberechts & Jesse Davis. ECML PKDD 2019.
  - Keywords: PU learning, Unlabeled data, Classification.
  - <details><summary>Digest</summary> We propose and theoretically analyze an empirical-risk-based method for incorporating the labeling mechanism. Additionally, we investigate under which assumptions learning is possible when the labeling mechanism is not fully understood and propose a practical method to enable this. 
  
- Large-Margin Label-Calibrated Support Vector Machines for Positive and Unlabeled Learning.[[paper]](
  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8636540)
  - Chen Gong, Tongliang Liu, Jian Yang, Dacheng Tao. IEEE Transactions on Neural Networks and Learning Systems 2019.
  - Keywords: Large-margin Label-calibrated Support Vector Machines (LLSVM), Positive and Unlabeled Learning.
  - <details><summary>Digest</summary> In this paper, we argue that the clusters formed by positive examples and potential negative examples in the feature space should be critically utilized to establish the PU learning model, especially when the negative data are not explicitly available. To this end, we introduce a hat loss to discover the margin between data clusters, a label calibration regularizer to amend the biased decision boundary to the potentially correct one, and propose a novel discriminative PU classifier termed “Large-margin Label-calibrated Support Vector Machines” (LLSVM). Our LLSVM classifier can work properly in the absence of negative training examples and effectively achieve the max-margin effect between positive and negative classes.
  
- Covariate Shift Adaptation on Learning from Positive and Unlabeled Data.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4411/4289)
  - Tomoya Sakai, Nobuyuki Shimizu. AAAI 2019.
  - Keywords: covariate shift, eweighted PU learning.
  - <details><summary>Digest</summary> We propose an importanceweighted PU learning method and reveal in which situations the importance-weighting is necessary. Moreover, we derive the convergence rate of the proposed method under mild conditions and experimentally demonstrate its effectiveness.
  
- Estimating the Class Prior in Positive and Unlabeled Data Through Decision Tree Induction. [[paper]](
  https://ojs.aaai.org/index.php/AAAI/article/view/11715/11574)
  - Jessa Bekker, Jesse Davis. AAAI 2018.
  - Keywords: Decision Tree Induction, Class Prior Estimation, Positive and Unlabeled Data.
  -  <details><summary>Digest</summary> In this paper, we propose a simple yet effective method for estimating the class prior, by estimating the probability that a positive example is selected to be labeled. Our key insight is that subdomains of the data give a lower bound on this probability. This lower bound gets closer to the real probability as the ratio of labeled examples increases. Finding such subsets can naturally be done via top-down decision tree induction. 
  
- Margin Based PU Learning.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11698/11557)
  - Tieliang Gong, Guangtao Wang, Jieping Ye, Zongben Xu and Ming Lin. AAAI 2018.
  - Keywords: PU Learning, Generalization error, Classification.
  -  <details><summary>Digest</summary>  In this work, we show that not all margin-based heuristic rules are able to improve the learned classifiers iteratively. We find that a so-called large positive margin oracle is necessary to guarantee the success of PU learning. Under this oracle, a provable positive-margin based PU learning algorithm is proposed for linear regression and classification under the truncated Gaussian distributions. The proposed algorithm is able to reduce the recovering error geometrically proportional to the positive margin. 

- Positive and Unlabeled Learning via Loss Decomposition and Centroid Estimation.[[paper]](https://www.ijcai.org/proceedings/2018/0373.pdf)
  - Hong Shi, Shaojun Pan, Jian Yang and Chen Gong. IJCAI 2018.
  - Keywords: Loss Decomposition and Centroid Estimation (LDCE).
  - <details><summary>Digest</summary> This paper regards all unlabeled examples as negative, which means that some of the original positive data are mistakenly labeled as negative. By doing so, we convert PU learning into the risk min- imization problem in the presence of false negative label noise, and propose a novel PU learning algo- rithm termed “Loss Decomposition and Centroid Estimation” (LDCE). By decomposing the hinge loss function into two parts, we show that only the second part is influenced by label noise, of which the adverse effect can be reduced by estimating the centroid of negative examples
  
- fBGD: Learning Embeddings From Positive Unlabeled Data with BGD. [[PAPER]](https://eprints.gla.ac.uk/166078/1/166078.pdf)
  - Fajie Yuan, Xin Xin, Xiangnan He, Guibing Guo, Weinan Zhang, Chua Tat-Seng and Joemon M. Jose. 2018.
  - Keywords: fast and generic batch gradient descent optimizer (fBGD).
  -  <details><summary>Digest</summary> We present a fast and generic batch gradient descent optimizer (fBGD) to learn from all training examples without sampling. By leveraging sparsity in PU data, we accelerate fBGD by several magnitudes, making its time complexity the same level as the NS-based stochastic gradient descent method. Meanwhile, we observe that the standard batch gradient method suffers from gradient instability issues due to the sparsity property.
  
- Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data.[[paper]](http://proceedings.mlr.press/v70/sakai17a/sakai17a.pdf)
  - Tomoya Sakai, Marthinus Christoffel Plessis, Gang Niu, Masashi Sugiyama. ICML 2017.
  - Keywords: classification, Semi-supervised learning.
  - <details><summary>Digest</summary> In this paper, we extend PU classification to also incorporate negative data and propose a novel semi-supervised learning approach. We establish generalization error bounds for our novel methods and show that the bounds decrease with respect to the number of unlabeled data without the distributional assumptions that are required in existing semi-supervised learning methods. 

- Positive-Unlabeled Learning with Non-Negative Risk Estimator.[[paper]](https://proceedings.neurips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)
  - Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, Masashi Sugiyama. NIPS 2017.
  - Keywords: Non-Negative Risk Estimator, Positive-Unlabeled Learning, overfitting.
  - <details><summary>Digest</summary> In this paper, we propose a non-negative risk estimator for PU learning: when getting minimized, it is more robust against overfitting, and thus we are able to use very flexible models (such as deep neural networks) given limited P data. Moreover, we analyze the bias, consistency, and mean-squared-error reduction of the proposed risk estimator, and bound the estimation error of the resulting empirical risk minimizer.
  
- Reconstruct & Crush Network. [[paper]](https://proceedings.neurips.cc/paper/2017/file/269d837afada308dd4aeab28ca2d57e4-Paper.pdf)
  - Erinc Merdivan, Mohammad Reza Loghmani, Matthieu Geist. NIPS 2017.
  - Keywords: energy-based model, Positive and Unlabeled (PU) learning, covariate shift, imbalanced data. 
  - <details><summary>Digest</summary> This article introduces an energy-based model that is adversarial regarding data: it minimizes the energy for a given data distribution (the positive samples) while maximizing the energy for another given data distribution (the negative or unlabeled samples). The model is especially instantiated with autoencoders where the energy, represented by the reconstruction error, provides a general distance measure for unknown data. The resulting neural network thus learns to reconstruct data from the first distribution while crushing data from the second distribution. This solution can handle different problems such as Positive and Unlabeled (PU) learning or covariate shift, especially with imbalanced data. Using autoencoders allows handling a large variety of data, such as images, text or even dialogues. 
  
- Multi-Positive and Unlabeled Learning.[[paper]](https://www.ijcai.org/proceedings/2017/0444.pdf)
  - Yixing Xu, Chang Xu, Chao Xu, Dacheng Tao. IJCAI 2017.
  - Keywords:  multi-class model.
  - <details><summary> Digest</summary> Here we propose a one-step method that directly enables multi-class model to be trained using the given input multi-class data and that predicts the label based on the model decision. Specifically, we construct different convex loss functions for labeled and unlabeled data to learn a discriminant function F. 
  
- Recovering True Classifier Performance in Positive-Unlabeled Learning.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/10937/10796)
  - Shantanu Jain, Martha White, Predrag Radivojac. AAAI 2017.
  - Keywords: ROC curve, AUC, Precision Recall curve, asymmetric noise, positive unlabeled learning, class prior estimation.
  - <details><summary> Digest</summary> In this work, we show that the typically used performance measures such as the receiver operating characteristic curve, or the precision recall curve obtained on such data can be corrected with the knowledge of class priors; i.e., the proportions of the positive and negative examples in the unlabeled data. We extend the results to a noisy setting where some of the examples labeled positive are in fact negative and show that the correction also requires the knowledge of the proportion of noisy examples in the labeled positives.
  
- Semi-supervised AUC optimization based on positive-unlabeled learning.[[paper]](https://link.springer.com/content/pdf/10.1007/s10994-017-5678-9.pdf?pdf=button)
  - Tomoya Sakai, Gang Niu & Masashi Sugiyama. Machine Learning 2017.
  - Keywords: AUC optimization, Semi-supervised learning, positive-unlabeled learning.
  - <details><summary> Digest</summary> In this paper, we propose a novel semi-supervised AUC optimization method that does not require such restrictive assumptions. We first develop an AUC optimization method based only on positive and unlabeled data and then extend it to semi-supervised learning by combining it with a supervised AUC optimization method. 

- Convex Formulation for Learning from Positive and Unlabeled Data.[[paper]](http://proceedings.mlr.press/v37/plessis15.pdf)
  - Marthinus Du Plessis, Gang Niu, Masashi Sugiyama. ICML 2015.
  - Keywords: convex formulation, double hinge loss. 
  - <details><summary>Digest</summary> In this paper, we discuss a convex formulation for PU classification that can still cancel the bias. The key idea is to use different loss functions for positive and unlabeled samples. However, in this setup, the hinge loss is not permissible. As an alternative, we propose the double hinge loss. 
  
- Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework [[paper]](https://www.researchgate.net/profile/Chen-Li-93/publication/269040485_Towards_Positive_Unlabeled_Learning_for_Parallel_Data_Mining_A_Random_Forest_Framework/links/54a9f99d0cf257a6360d5b7f/Towards-Positive-Unlabeled-Learning-for-Parallel-Data-Mining-A-Random-Forest-Framework.pdf)
  - Chen Li and Xue-Liang Hua. ADMA 2014.
  - Keywords: PU information gain, PU Gini index, random forest, parallel data mining.
  - <details><summary>Digest</summary> We investigate widely used Positive and Unlabeled (PU) learning algorithms including PU information gain and a newly developed PU Gini index combining with popular parallel computing framework - Random Forest (RF), thereby enabling parallel data mining to learn from only positive and unlabeled samples. The proposed framework, termed PURF (Positive Un- labeled Random Forest), is able to learn from positive and unlabeled instances and achieve comparable classifcation performance with RF trained by fully la- beled data through parallel computing according to experiments on both synthetic and real-world UCI datasets. 
  
- A bagging SVM to learn from positive and unlabeled examples. [[paper]](https://www.sciencedirect.com/science/article/pii/S0167865513002432/pdfft?md5=4e9937f1cc1315c94b40e9122b83bc54&pid=1-s2.0-S0167865513002432-main.pdf)
  - F.Mordelet, J.-P.Vert. Pattern Recognition Letters 2014.
  - Keywords: PU bagging.
  - <details><summary>Digest</summary> We propose a new method for PU learning with a conceptually simple implementation based on bootstrap aggregating (bagging) techniques: the algorithm iteratively trains many binary classifiers to discriminate the known positive examples from random subsamples of the unlabeled set, and averages their predictions.
  
  
- Distributional Similarity vs. PU Learning for Entity Set Expansion. [[paper]](https://aclanthology.org/P10-2066.pdf)
  - Xiao-Li Li, Lei Zhang, Bing Liu, See-Kiong Ng. ACL 2010.
  - Keywords: Distributional Similarity, PU Learning, Entity Set Expansion.
  - <details><summary>Digest</summary> Distributional similarity is a classic tech- nique for entity set expansion, where the system is given a set of seed entities of a particular class, and is asked to expand the set using a corpus to obtain more entities of the same class as represented by the seeds. This paper shows that a machine learning model called positive and unlabeled learning (PU learning) can model the set expansion problem better.
  
- Learning from Positive and Unlabeled Examples with Different Data Distributions.[[paper]](https://link.springer.com/content/pdf/10.1007/11564096_24.pdf?pdf=inline%20link)
  - Xiao-Li Li and Bing Liu. ECML 2005.
  - Keywords: Different Data Distributions, A-EM.
  - <details><summary>Digest</summary> We study the problem of learning from positive and unlabeled examples. Although several techniques exist for dealing with this problem, they all assume that positive examples in the positive set P and the positive examples in the unlabeled set U are generated from the same distribution. This assumption may be violated in practice. This paper proposes a novel technique A-EM to deal with the problem. Experiment results with product page classification demonstrate the effectiveness of the proposed technique.
  
## Application Paper in Medical Image Analysis

- ShapePU: A New PU Learning Framework Regularized by Global Consistency for Scribble Supervised Cardiac Segmentation[[paper]](https://arxiv.org/pdf/2206.02118.pdf)[[code]](https://github.com/BWGZK/ShapePU)
  - Ke Zhang and Xiahai Zhuang. MICCAI 2022.
  - Keywords: Weakly supervised learning, PU learning, Segmentation.
  - Application: Cardiac segmentation.
  - <details><summary>Digest</summary> We propose a new scribble-guided method for cardiac segmentation, based on the Positive-Unlabeled (PU) learning framework and global consistency regularization, and termed as ShapePU. To leverage unlabeled pixels via PU learning, we first present an Expectation-Maximization (EM) algorithm to estimate the proportion of each class in the unlabeled pixels. Given the estimated ratios, we then introduce the marginal probability maximization to identify the classes of unlabeled pixels. To exploit shape knowledge, we apply cutout operations to training images, and penalize the inconsistent segmentation results. 
  
- Anatomy-Guided Weakly-Supervised Abnormality Localization in Chest X-rays. [[paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-16443-9.pdf?pdf=button)[[code]](https://github.com/batmanlab/AGXNet)
  - Ke Yu, Shantanu Ghosh, Zhexiong Liu, Christopher Deible & Kayhan Batmanghelich. MICCAI 2022.
  - Keywords: Weakly-supervised learning, PU learning, Disease detection, Class activation map, Residual attention.
  - Application: Anatomy-Guided chest X-ray Network to address these issues of weak annotation.
  - <details><summary>Digest</summary> We propose an Anatomy-Guided chest X-ray Network (AGXNet) to address these issues of weak annotation. Our framework consists of a cascade of two networks, one responsible for identifying anatomical abnormalities and the second responsible for pathological observations. The critical component in our framework is an anatomy-guided attention module that aids the downstream observation network in focusing on the relevant anatomical regions generated by the anatomy network. We use Positive Unlabeled (PU) learning to account for the fact that lack of mention does not necessarily mean a negative label. Our quantitative and qualitative results on the MIMIC-CXR dataset demonstrate the effectiveness of AGXNet in disease and anatomical abnormality localization.
  
- Embryo Grading With Unreliable L abels Due to Chromosome Abnormalities by Regularized PU Learning With Ranking. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9606688&tag=1)
  - Masashi Nagaya and Norimichi Ukita. IEEE Transactions on Medical Imaging 2022.
  - Keywords: Deep convolutional networks, positive-unlabeled learning, learning-to-rank, mutual information.
  - Application: human embryo grading with its images.
  - <details><summary>Digest</summary> For alleviating an adverse effect of the unreliable labels, our method employs Positive-Unlabeled (PU) learning so that live birth and non-live birth are labeled as positive and unlabeled, respectively, where unlabeled samples contain both positive and negative samples. In our method, this PU learning on a deep CNN is improved by a learning-to-rank scheme. While the original learning-to- rank scheme is designed for positive-negative learning, it is extended to PU learning. Furthermore, overfitting in this PU learning is alleviated by regularization with mutual information.
  
- Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training. [[paper]](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf)
  - Xuxi Chen, Wuyang Chen, Tianlong Chen, Ye Yuan, Chen Gong, Kewei Chen, Zhangyang Wang. ICML 2020.
  - Application: classifying brain images of Alzheimer’s Disease.
  - <details><summary>Digest</summary> This paper proposed a novel Self-PU learning framework, which seamlessly integrates PU learning and self-training. Self-PU highlights three “self”-oriented building blocks: a self-paced training algorithm that adaptively discovers and augments confident positive/negative examples as the training proceeds; a self-reweighted, instance-aware loss; and a self-distillation scheme that introduces teacher-students learning as an effective regularization for PU learning. 

  
- Positive-Unlabeled Learning for Cell Detection in Histopathology Images with Incomplete Annotations [[paper]](https://arxiv.org/pdf/2106.15918.pdf)
  - Zipei Zhao, Fengqian Pang, Zhiwen Liu, Chuyang Ye. MICCAI 2021.
  - Keywords: Cell detection, positive-unlabeled learning, incomplete annotation.
  - Application: Cell detection in histopathology images.
  - <details><summary>Digest</summary>  In this work, to address the problem of incomplete annotations, we formulate the training of detection networks as a positive-unlabeled learning problem. Specifically, the classification loss in network training is revised to take into account incomplete annotations, where the terms corresponding to negative samples are approximated with the true positive samples and the other sam- ples of which the labels are unknown.

- Semi-Supervised Screening of COVID-19 from Positive and Unlabeled Data with Constraint Non-Negative Risk Estimator.[[paper]](https://link.springer.com/content/pdf/10.1007/978-3-030-78191-0.pdf?pdf=button)
  - Zhongyi Han, Rundong He, Tianyang Li, Benzheng Wei, Jian Wang & Yilong Yin. IPMI 2021.
  - Keywords: COVID-19 screening, Positive unlabled learning, X-ray, computed tomography.
  - <details><summary>Digest</summary> We propose a new PU learning method called Constraint Non-Negative Positive Unlabeled Learning (cnPU). It suggests the constraint non-negative risk estimator, which is more robust against overfitting than previous PU learning methods when giving limited positive data. It also embodies a new and efficient optimization algorithm that can make the model learn well on positive data and avoid overfitting on unlabeled data. To the best of our knowledge, this is the first work that realizes PU learning of COVID-19.
  
- Cell Detection from Imperfect Annotation by Pseudo Label Selection Using P-classification. [[paper]](https://link.springer.com/content/pdf/10.1007/978-3-030-87237-3.pdf?pdf=button)
  - Keywords: Cell detection, Imperfect annotation.
  - Application: Cell detection.
  - <details><summary>Digest</summary> Our proposed method takes a pseudo labeling approach for cell detection from imperfect annotated data. A detection convolutional neural network (CNN) trained using such missing labeled data often produces over-detection. We treat partially labeled cells as positive samples and the detected positions except for the labeled cell as unlabeled samples. Then we select reliable pseudo labels from unlabeled data using recent machine learning techniques; positive-and-unlabeled (PU) learning and P-classification.
  
- 3D-BoxSup: Positive-Unlabeled Learning of Brain Tumor Segmentation Networks From 3D Bounding Boxes. [[paper]](https://www.frontiersin.org/articles/10.3389/fnins.2020.00350/full)
  - Yanwu Xu, Mingming Gong, Junxiang Chen, Ziye Chen and Kayhan Batmanghelich. Frontiers in Neuroscience 2020.
  - Keywords: 3D Bounding Boxes, 3D-BoxSup, Brain Tumor Segmentation.
  - Application: Brain tumor segmentation.
  - <details><summary>Digest</summary> In this paper, we have proposed a method that achieves competitive accuracy from a “weakly annotated” image where the weak annotation is obtained via a 3D bounding box denoting an object of interest. Our method, called “3D-BoxSup,” employs a positive-unlabeled learning framework to learn segmentation masks from 3D bounding boxes. Specially, we consider the pixels outside of the bounding box as positively labeled data and the pixels inside the bounding box as unlabeled data. Our method can suppress the negative effects of pixels residing between the true segmentation mask and the 3D bounding box and produce accurate segmentation masks. We applied our method to segment a brain tumor. 
  
- Negative-Unlabeled Learning for Diffusion MRI. [[paper]](https://vision.cs.tum.edu/_media/spezial/bib/swazinna-et-al-ismrm2019.pdf)
  - Phillip Swazinna, Vladimir Golkov, Ilona Lipp, Eleonora Sgarlata, Valentina Tomassini, Derek K. Jones, and Daniel Cremers. ISMRM 2019.
  - Keywords: Negative-Unlabeled Learning.
  - Application: MRI.
  - <details><summary>Digest</summary> Machine learning strongly enhances diffusion MRI in terms of acquisition speed and quality of results. Different machine learning tasks are applicable in different situations: labels for training might be available only for healthy data or only for common but not rare diseases; training labels might be available voxel-wise, or only scan-wise. This leads to various tasks beyond supervised learning. Here we examine whether it is possible to perform accurate voxel-wise MS lesion detection if only scan-wise training labels are used.

  
- Deep positive-unlabeled learning for region of interest localization in breast tissue images.[[paper]](https://doi.org/10.1117/12.2293721)
  - Pushpak Pati, Sonali Andani, Matthew Pediaditis, Matheus Palhares Viana, Jan Hendrik Rüschoff, Peter Wild, Maria Gabrani. IPMI 2018.
  - Keywords: tumor-region-of-interest (TRoI), breast tissue images, high-power-fields (HPFs).
  - Application: detection of tumor-region-of-interest (TRoI) on WSIs of breast tissue.
  - <details><summary>Digest</summary> In this work, we propose a positive and unlabeled learning approach that uses a few examples of HPF regions (positive annotations) to localize the invasive TRoIs on breast cancer WSIs. We use unsupervised deep autoencoders with Gaussian Mixture Model-based clustering to identify the TRoI in a patch-wise manner. The algorithm is developed using 90 HPF-annotated WSIs and is validated on 30 fully-annotated WSIs.
  
- Learning from Only Positive and Unlabeled Data to Detect Lesions in Vascular CT Images.[[paper]](https://link.springer.com/content/pdf/10.1007/978-3-642-23626-6.pdf?pdf=button)
  - Maria A. Zuluaga, Don Hush, Edgar J. F. Delgado Leyton, Marcela Hernández Hoyos & Maciej Orkisz. MICCAI 2011.
  - Keywords: Support Vector Machine, Synthetic Data, True Positive Rate, Unlabeled Data, Empirical Risk.
  - <details><summary>Digest</summary> While most existing solutions tackle calcified and non-calcified plaques separately, we present a new algorithm capable of detecting both types of lesions in CT images. It builds up on a semi-supervised classification framework, in which the training set is made of both unlabeled data and a small amount of data labeled as normal. Our method takes advantage of the arrival of newly acquired data to re-train the classifier and improve its performance. 
  
  
  
