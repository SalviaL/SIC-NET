# SIC-NET

Spatial Information Considered Network for Scene Classification

This paper has been published on *IEEE Geoscience and Remote Sensing Letters*
If you want to use this code, cite as follow please:

```
@ARTICLE{9094628,
  author={C. {Tao} and W. {Lu} and J. {Qi} and H. {Wang}},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Spatial Information Considered Network for Scene Classification}, 
  year={2020},
  volume={},
  number={},
  pages={1-5},}
```

## How to use

`model = model_sic_net(512, 7, 3, 3).cuda()` means creating a model as follow:

```flow
st=>start: Encoder feature map
CTU1=>operation: CTU
CTU2=>operation: CTU
CTU3=>operation: CTU
FCN1=>operation: FCN
FCN2=>operation: FCN
FCN3=>operation: FCN
e=>end: Result
st->CTU1->CTU2->CTU3->FCN1->FCN2->FCN3->e
```
