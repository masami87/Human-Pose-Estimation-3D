# NTU说明

## 骨架
该数据集包含56880个数据样本，一共60类动作，前面50类动作是单人动作，后面10类动作是双人交互动作，数据集样本文件都是例如这样的格式：S001C003P008R002A058.skeleton。
S:设置号，“ NTU RGB + D”数据集包括设置号在S001和S017之间的文件/文件夹，而“ NTU RGB + D 120”数据集包括设置号在S001和S032之间的文件/文件夹。
C:相机ID，共有三架，
P:人物ID，P001表示一号动作执行人，但并非每个人都执行了所有动作，
R:同一个动作的表演次数，
A:动作类别，A001到A060这么多动作类别。

对于单个.skeleton文件：
```txt
70   # 单个样本包含的总帧数 
2     #从第二行开始分别为每一帧的信息,对于其中每一帧，第一个数字为当前帧body数量（如1或2）
72057594037944738 0 1 1 1 1 0 0.01119184 -0.256052 2  
# body_info =[ 'bodyID', 'clipedEdges', 'handLeftConfidence',
#                  'handLeftState', 'handRightConfidence', 'handRightState',
#                 'isResticted', 'leanX', 'leanY', 'trackingState' ]*/
25    # 25个关节点 25*12  每个关节包含12个字段信息
# joint_info_key = [
#                     'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
#                     'orientationW', 'orientationX', 'orientationY',
#                     'orientationZ', 'trackingState'
#                 ]
0.7201789 0.1647426 3.561895 336.5812 192.8021 1184.537 489.4118 -0.2218947 0.03421978 0.968875 -0.1042736 2
0.7260004 0.4483242 3.498769 338.6059 162.7653 1190.205 402.5223 -0.2523348 0.03749071 0.9613965 -0.1031427 2
0.7291254 0.7261137 3.425736 340.6862 131.9133 1195.987 313.4265 -0.2830125 0.04291317 0.9484802 -0.135822 2
0.7023438 0.8463342 3.396255 338.5144 118.1859 1189.627 273.8169 0 0 0 0 2
0.5827141 0.6567038 3.536651 322.9463 141.66 1144.507 341.585 0.2607923 0.6977255 -0.6554007 -0.124966 2
0.5192263 0.3914464 3.649472 314.598 170.4908 1120.394 424.959 -0.03581179 -0.6068473 0.2222077 0.7622845 2
0.4511961 0.1653868 3.66607 307.5214 193.2507 1100.169 490.9451 0.12472 0.7260504 -0.0826719 0.671164 2
0.4417633 0.1185798 3.647333 306.8019 197.8581 1098.219 504.3104 -0.06916566 0.7038938 -0.1973353 0.6788287 2
0.8548245 0.6050068 3.364435 355.8548 143.6936 1240.168 347.399 -0.1105671 0.7569718 0.5679373 -0.3036705 2
0.9127819 0.3429818 3.401966 361.0178 172.7352 1255.312 431.2415 0.05843762 0.9865138 0.1144138 0.1014157 2
0.8905535 0.1046963 3.41965 358.0569 198.5103 1247.062 505.7414 0.05872061 0.7232746 -0.008176846 0.6880109 2
0.87075 0.04412444 3.425586 355.7503 205.0164 1240.474 524.5703 0.1441216 0.6976146 -0.07766516 0.6975178 2
0.6528355 0.1655106 3.560884 329.6426 192.7268 1164.535 489.2483 -0.07249752 -0.6982451 0.7074214 -0.08217493 2
0.6853271 -0.132269 3.735053 329.6935 222.7012 1164.244 576.0424 -0.178642 -0.5079327 0.195941 0.8195722 2
0.7497366 -0.4176336 3.852802 333.831 249.4536 1175.846 653.3472 -0.184775 -0.4768907 0.115771 0.8514872 2
0.6873895 -0.4799173 3.752486 329.6509 256.5972 1164.204 674.0577 0 0 0 0 2
0.7747468 0.1611266 3.501125 343.617 192.8774 1205.077 489.5748 -0.2383845 0.6385497 0.6488895 -0.3381858 2
0.8171204 -0.1421985 3.566106 346.5092 224.3531 1213.413 580.5895 0.0535739 0.8461024 0.1127898 0.5181884 2
0.8511656 -0.4595799 3.664586 347.7344 255.7416 1216.645 671.2434 0.1045282 0.8594967 0.1184092 0.4861262 2
0.7877548 -0.521389 3.564552 343.6025 263.3947 1205.197 693.4037 0 0 0 0 2
0.728802 0.6575422 3.445887 340.162 139.7047 1194.523 335.9147 -0.2832572 0.0415111 0.9510412 -0.1164597 2
0.4375839 0.07206824 3.637431 306.4994 202.4994 1097.437 517.7725 0 0 0 0 2
0.4698927 0.112225 3.619 309.9973 198.4049 1107.594 505.8666 0 0 0 0 2
0.8659531 -0.02500954 3.4443 354.7251 212.4024 1237.508 545.9329 0 0 0 0 2
0.901369 0.03076283 3.417449 359.2839 206.438 1250.683 528.6425 0 0 0 0 2
72057594037944734 0 1 1 0 0 0 0.2471961 -0.2383252 2
25
-0.387201 -0.05403496 2.961099 214.7408 216.41 834.7666 559.0992 0.4557798 -0.05701343 0.8802808 -0.1188276 2
-0.384766 0.2418743 2.881034 213.6927 179.0352 831.9214 450.6993 0.4476836 -0.05592072 0.8844258 -0.1193457 2
-0.3798909 0.5306301 2.789107 212.6189 140.0148 829.2738 337.7772 0.4369697 -0.06553705 0.8833419 -0.1564273 2
-0.3326302 0.6499826 2.820972 219.2917 125.236 848.3054 294.9901 0 0 0 0 2
-0.4304172 0.3858192 2.693917 204.0139 157.2809 805.1432 387.8424 -0.2866399 0.7234381 -0.4137073 0.4725688 2
-0.4339336 0.1272796 2.714383 204.0563 192.5867 805.3087 490.1169 0.03946908 0.9411238 0.006875069 0.3356799 2
-0.36822 -0.07224102 2.786263 214.2289 219.2174 834.5075 567.2477 0.2250257 0.7364373 0.00601962 -0.6379557 2
-0.3700575 -0.127083 2.808011 214.3589 226.2849 834.8156 587.7428 0.1314776 0.7426878 -0.1347235 -0.6426337 2
-0.2981097 0.4353881 2.893557 224.8335 154.6743 863.9844 379.9771 0.2839388 0.7432098 0.5627881 0.2242489 1
-0.2671633 0.2557262 3.116553 231.2118 179.7559 881.2538 452.5558 0.2853988 0.6805304 0.3266041 0.5905555 1
-0.2581481 -0.01978063 3.185815 232.9456 212.0073 886.1906 546.1407 0.122517 0.9550178 -0.01796174 -0.2694587 1
-0.229458 -0.05952364 3.223031 236.5407 216.4835 896.474 559.0983 0.402256 0.8494924 0.1237116 -0.318195 1
-0.4174743 -0.05169832 2.87393 209.4239 216.3172 819.9359 558.8809 0.3790196 -0.5433965 0.6074547 -0.4382502 2
-0.497618 -0.373121 2.919716 200.112 256.5469 793.2654 675.4592 0.03238185 0.950422 -0.1355495 -0.2779852 2
-0.5771378 -0.6777439 2.980165 191.4323 293.2393 768.4659 781.3873 0.05410571 0.9433935 -0.1481987 -0.2917506 2
-0.5500029 -0.756217 2.962994 194.3529 303.4882 777.1974 810.9081 0 0 0 0 2
-0.3476626 -0.05360388 2.980002 219.9058 216.3129 849.6191 558.7677 0.2399694 0.7000011 0.6688294 0.07127756 2
-0.3613364 -0.3393445 3.148574 220.5649 249.1572 850.9557 653.9677 0.1018645 0.298928 0.2436071 0.9170176 2
-0.3918813 -0.6122444 3.258895 218.4803 278.5703 844.7856 738.9781 0.1042677 0.2772751 0.1674781 0.9403179 2
-0.3447265 -0.6813605 3.248184 223.6403 286.6198 859.91 762.1609 0 0 0 0 2
-0.3814697 0.4595191 2.81428 212.8908 149.9362 829.9158 366.4533 0.4385106 -0.05917181 0.8863857 -0.1361158 2
-0.3481399 -0.1767624 2.830068 217.567 232.5726 844.04 605.946 0 0 0 0 2
-0.4126344 -0.1371371 2.798348 208.6103 227.6621 818.2493 591.783 0 0 0 0 2
-0.2210823 -0.06690916 3.229667 237.5418 217.3045 899.3542 561.4713 0 0 0 0 1
-0.2278395 -0.05910378 3.200297 236.5407 216.4835 896.598 559.0983 0 0 0 0 1
```

## 深度图
与S001C003P008R002A058.skeleton文件对应的深度图目录下S001C003P008R002A058文件夹，其中包含每一帧的深度图如：MDepth-00000067.png。