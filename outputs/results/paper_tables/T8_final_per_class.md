Table T8. Per-class test metrics of the final model with bootstrap intervals for F1.

| class | name | group | precision | recall | f1 | f1_ci_low | f1_ci_high | support_messages | support_vehicles |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Genuine | genuine | 0.9710 | 0.9998 | 0.9852 | 0.9835 | 0.9868 | 288005 | 2589 |
| 1 | Constant position | fault | 0.9901 | 0.9942 | 0.9922 | 0.9864 | 0.9955 | 6246 | 59 |
| 2 | Constant position offset | fault | 0.9937 | 0.8702 | 0.9279 | 0.9103 | 0.9434 | 6566 | 59 |
| 3 | Random position | fault | 0.9893 | 0.9904 | 0.9899 | 0.9863 | 0.9925 | 5527 | 59 |
| 4 | Random position offset | fault | 0.9908 | 0.9951 | 0.9929 | 0.9901 | 0.9951 | 6924 | 59 |
| 5 | Constant speed | fault | 0.9946 | 0.9943 | 0.9944 | 0.9932 | 0.9955 | 6103 | 59 |
| 6 | Constant speed offset | fault | 0.9967 | 0.9991 | 0.9979 | 0.9970 | 0.9987 | 6924 | 59 |
| 7 | Random speed | fault | 0.9944 | 0.9947 | 0.9945 | 0.9931 | 0.9957 | 6752 | 59 |
| 8 | Random speed offset | fault | 0.9981 | 0.9964 | 0.9972 | 0.9962 | 0.9982 | 6887 | 59 |
| 9 | Eventual stop | attack | 0.9996 | 0.8150 | 0.8979 | 0.8618 | 0.9247 | 5995 | 59 |
| 10 | Disruptive | attack | 0.8604 | 0.8076 | 0.8332 | 0.7882 | 0.8736 | 5526 | 59 |
| 11 | Data replay | attack | 0.8524 | 0.8381 | 0.8452 | 0.8143 | 0.8730 | 6098 | 59 |
| 12 | Delayed messages | fault | 0.9865 | 0.6881 | 0.8107 | 0.7694 | 0.8553 | 6037 | 59 |
| 13 | DoS | attack | 0.9722 | 0.9966 | 0.9843 | 0.9711 | 0.9933 | 18333 | 59 |
| 14 | DoS random | attack | 0.9061 | 0.9843 | 0.9436 | 0.9160 | 0.9675 | 19823 | 59 |
| 15 | DoS disruptive | attack | 0.9851 | 0.8707 | 0.9244 | 0.8922 | 0.9532 | 19351 | 59 |
| 16 | Grid Sybil | attack | 0.9923 | 0.8298 | 0.9038 | 0.8845 | 0.9197 | 24390 | 59 |
| 17 | Data replay Sybil | attack | 0.7850 | 0.5314 | 0.6338 | 0.6017 | 0.6616 | 6547 | 59 |
| 18 | DoS random Sybil | attack | 0.9032 | 0.9719 | 0.9363 | 0.9117 | 0.9563 | 13144 | 59 |
| 19 | DoS disruptive Sybil | attack | 0.7964 | 0.8429 | 0.8190 | 0.7881 | 0.8456 | 13008 | 59 |
