# Drug3D-Net
A Spatial-temporal Gated Attention Module for Molecular Property Prediction Based on Molecular Geometry. This is the official code implementation of Drug3D-Net paper. But the algorithm has been optimized and improved, which is slightly different from the original version.

<b>Requirements</b> <br>
> Linux (We only tested on Ubuntu-16.04)<br>
> Keras (version == 2.3.1)<br>
> Python (version == 3.6.8)<br>
> tensorflow (version == 1.15.0)<br>
> matplotlib (version == 3.3.2)<br>
> numpy (version == 1.19.4)<br>
> scikit-learn (version == 0.23.2)<br>
> scipy (version == 1.5.4)<br>
> pandas (version ==1.1.5)<br>

<b>DataSets</b> <br>
> esol_v2.csv<br>
> FreeSolv_SAMPL.csv<br>
> HIV.csv<br>
> tox21.csv<br>

<b>PreprocessData</b> <br>
> grid3Dmols_delaney<br>
> grid3Dmols_freesolv<br>
> grid3Dmols_hiv<br>
> tox21<br>
>> grid3Dmols_tox21_NR_AhR<br>
>> grid3Dmols_tox21_NR-AR<br>
>> grid3Dmols_tox21_NR-AR-LBD<br>
>> grid3Dmols_tox21_NR-Aromatase<br>
>> grid3Dmols_tox21_NR-ER<br>
>> grid3Dmols_tox21_NR-ER-LBD<br>
>> grid3Dmols_tox21_NR-PPAR-gamma<br>
>> grid3Dmols_tox21_SR-ARE<br>
>> grid3Dmols_tox21_SR-ATAD5<br>
>> grid3Dmols_tox21_SR-HSE<br>
>> grid3Dmols_tox21_SR-MMP<br>
>> grid3Dmols_tox21_SR-p53<br>

<b>SaveModel (Take the HIV dataset as an example)</b> <br>
> 6-0.979.hdf5<br>
> best_weight.hdf5<br>
> events.out.tfevents.1608729525.lab406<br>
> hyper.csv<br>
> raw_results.csv<br>

## Citation
Chunyan Li, Jianmin Wang, Zhangming Niu, Junfeng Yao, Xiangxiang Zeng, A spatial-temporal gated attention module for molecular property prediction based on molecular geometry, Briefings in Bioinformatics, 2021;, bbab078
https://doi.org/10.1093/bib/bbab078
