\ENCODING=ISO-8859-1
\Problem name: ilog.cplex

Minimize
 obj: _64step1#16 + _64step2#17 + _64step3#18 + _65step1#39 + _65step2#40
      + _65step3#41 + _66step1#50 + _66step2#51 + _66step3#52 + _56step1#59
      + _56step2#60 + _56step3#61 + _73step1#82 + _73step2#83 + _73step3#84
      + _43step1#100 + _43step2#101 + _43step3#102 + _63step1#108
      + _63step2#109 + _63step3#110
Subject To
 LB_64T16#0:                   _64T16#0 >= 80
 LB_64T15#1:                   _64T15#1 >= 582
 prec_64T15_64T16#2:           - _64T16#0 + _64T15#1 >= 502
 LB_64T14#3:                   _64T14#2 >= 683
 prec_64T14_64T15#4:           - _64T15#1 + _64T14#2 >= 101
 LB_64T13#5:                   _64T13#3 >= 891
 prec_64T13_64T14#6:           - _64T14#2 + _64T13#3 >= 208
 LB_64T12#7:                   _64T12#4 >= 995
 prec_64T12_64T13#8:           - _64T13#3 + _64T12#4 >= 104
 LB_64T11#9:                   _64T11#5 >= 1150
 prec_64T11_64T12#10:          - _64T12#4 + _64T11#5 >= 155
 LB_64T10#11:                  _64T10#6 >= 1557
 prec_64T10_64T11#12:          - _64T11#5 + _64T10#6 >= 407
 LB_64T9#13:                   _64T9#7 >= 1845
 prec_64T9_64T10#14:           - _64T10#6 + _64T9#7 >= 288
 LB_64T8#15:                   _64T8#8 >= 2780
 prec_64T8_64T9#16:            - _64T9#7 + _64T8#8 >= 935
 LB_64T7#17:                   _64T7#9 >= 3550
 prec_64T7_64T8#18:            - _64T8#8 + _64T7#9 >= 770
 LB_64T6#19:                   _64T6#10 >= 3791
 prec_64T6_64T7#20:            - _64T7#9 + _64T6#10 >= 241
 LB_64T5#21:                   _64T5#11 >= 4700
 prec_64T5_64T6#22:            - _64T6#10 + _64T5#11 >= 909
 LB_64T4#23:                   _64T4#12 >= 5313
 prec_64T4_64T5#24:            - _64T5#11 + _64T4#12 >= 613
 LB_64T3#25:                   _64T3#13 >= 5579
 prec_64T3_64T4#26:            - _64T4#12 + _64T3#13 >= 266
 LB_64T2#27:                   _64T2#14 >= 6059
 prec_64T2_64T3#28:            - _64T3#13 + _64T2#14 >= 480
 LB_64T1#29:                   _64T1#15 >= 6274
 prec_64T1_64T2#30:            - _64T2#14 + _64T1#15 >= 215
 _64step1#31:                  _64T1#15 - 75318 _64step1#16 <= 6194
 _64step2#32:                  _64T1#15 - 75138 _64step2#17 <= 6374
 _64step3#33:                  _64T1#15 - 74958 _64step3#18 <= 6554
 LB_65T25#34:                  _65T25#19 >= 191
 LB_65T24#35:                  _65T24#20 >= 408
 prec_65T24_65T25#36:          - _65T25#19 + _65T24#20 >= 217
 LB_65T23#37:                  _65T23#21 >= 581
 prec_65T23_65T24#38:          - _65T24#20 + _65T23#21 >= 173
 LB_65T22#39:                  _65T22#22 >= 821
 prec_65T22_65T23#40:          - _65T23#21 + _65T22#22 >= 240
 LB_65T21#41:                  _65T21#23 >= 1001
 prec_65T21_65T22#42:          - _65T22#22 + _65T21#23 >= 180
 LB_65T20#43:                  _65T20#24 >= 1151
 prec_65T20_65T21#44:          - _65T21#23 + _65T20#24 >= 150
 LB_65T19#45:                  _65T19#25 >= 1196
 prec_65T19_65T20#46:          - _65T20#24 + _65T19#25 >= 45
 LB_65T18#47:                  _65T18#26 >= 1451
 prec_65T18_65T19#48:          - _65T19#25 + _65T18#26 >= 255
 LB_65T17#49:                  _65T17#27 >= 1781
 prec_65T17_65T18#50:          - _65T18#26 + _65T17#27 >= 330
 LB_65T16#51:                  _65T16#28 >= 2021
 prec_65T16_65T17#52:          - _65T17#27 + _65T16#28 >= 240
 LB_65T15#53:                  _65T15#29 >= 2261
 prec_65T15_65T16#54:          - _65T16#28 + _65T15#29 >= 240
 LB_65T14#55:                  _65T14#30 >= 2441
 prec_65T14_65T15#56:          - _65T15#29 + _65T14#30 >= 180
 LB_65T13#57:                  _65T13#31 >= 2642
 prec_65T13_65T14#58:          - _65T14#30 + _65T13#31 >= 201
 LB_65T12#59:                  _65T12#32 >= 2726
 prec_65T12_65T13#60:          - _65T13#31 + _65T12#32 >= 84
 LB_65T11#61:                  _65T11#33 >= 2931
 prec_65T11_65T12#62:          - _65T12#32 + _65T11#33 >= 205
 LB_65T10#63:                  _65T10#34 >= 3811
 prec_65T10_65T11#64:          - _65T11#33 + _65T10#34 >= 880
 LB_65T9#65:                   _65T9#35 >= 4231
 prec_65T9_65T10#66:           - _65T10#34 + _65T9#35 >= 420
 LB_65T8#67:                   _65T8#36 >= 4501
 prec_65T8_65T9#68:            - _65T9#35 + _65T8#36 >= 270
 LB_65T7#69:                   _65T7#37 >= 4771
 prec_65T7_65T8#70:            - _65T8#36 + _65T7#37 >= 270
 LB_65T6#71:                   _65T6#38 >= 4951
 prec_65T6_65T7#72:            - _65T7#37 + _65T6#38 >= 180
 _65step1#73:                  _65T6#38 - 75318 _65step1#39 <= 4760
 _65step2#74:                  _65T6#38 - 75138 _65step2#40 <= 4940
 _65step3#75:                  _65T6#38 - 74958 _65step3#41 <= 5120
 LB_66T18#76:                  _66T18#42 >= 1
 LB_66T19#77:                  _66T19#43 >= 421
 prec_66T19_66T18#78:          - _66T18#42 + _66T19#43 >= 420
 LB_66T20#79:                  _66T20#44 >= 511
 prec_66T20_66T19#80:          - _66T19#43 + _66T20#44 >= 90
 LB_66T21#81:                  _66T21#45 >= 541
 prec_66T21_66T20#82:          - _66T20#44 + _66T21#45 >= 30
 LB_66T22#83:                  _66T22#46 >= 721
 prec_66T22_66T21#84:          - _66T21#45 + _66T22#46 >= 180
 LB_66T23#85:                  _66T23#47 >= 841
 prec_66T23_66T22#86:          - _66T22#46 + _66T23#47 >= 120
 LB_66T24#87:                  _66T24#48 >= 961
 prec_66T24_66T23#88:          - _66T23#47 + _66T24#48 >= 120
 LB_66T25#89:                  _66T25#49 >= 1071
 prec_66T25_66T24#90:          - _66T24#48 + _66T25#49 >= 110
 _66step1#91:                  _66T25#49 - 75318 _66step1#50 <= 1070
 _66step2#92:                  _66T25#49 - 75138 _66step2#51 <= 1250
 _66step3#93:                  _66T25#49 - 74958 _66step3#52 <= 1430
 LB_56T11#94:                  _56T11#53 >= 1
 LB_56T10#95:                  _56T10#54 >= 306
 prec_56T10_56T11#96:          - _56T11#53 + _56T10#54 >= 305
 LB_56T9#97:                   _56T9#55 >= 511
 prec_56T9_56T10#98:           - _56T10#54 + _56T9#55 >= 205
 LB_56T8#99:                   _56T8#56 >= 781
 prec_56T8_56T9#100:           - _56T9#55 + _56T8#56 >= 270
 LB_56T7#101:                  _56T7#57 >= 1103
 prec_56T7_56T8#102:           - _56T8#56 + _56T7#57 >= 322
 LB_56T6#103:                  _56T6#58 >= 1651
 prec_56T6_56T7#104:           - _56T7#57 + _56T6#58 >= 548
 _56step1#105:                 _56T6#58 - 75318 _56step1#59 <= 1650
 _56step2#106:                 _56T6#58 - 75138 _56step2#60 <= 1830
 _56step3#107:                 _56T6#58 - 74958 _56step3#61 <= 2010
 LB_73T25#108:                 _73T25#62 >= 1260
 LB_73T24#109:                 _73T24#63 >= 1590
 prec_73T24_73T25#110:         - _73T25#62 + _73T24#63 >= 330
 LB_73T23#111:                 _73T23#64 >= 1830
 prec_73T23_73T24#112:         - _73T24#63 + _73T23#64 >= 240
 LB_73T22#113:                 _73T22#65 >= 2070
 prec_73T22_73T23#114:         - _73T23#64 + _73T22#65 >= 240
 LB_73T21#115:                 _73T21#66 >= 2190
 prec_73T21_73T22#116:         - _73T22#65 + _73T21#66 >= 120
 LB_73T20#117:                 _73T20#67 >= 2370
 prec_73T20_73T21#118:         - _73T21#66 + _73T20#67 >= 180
 LB_73T19#119:                 _73T19#68 >= 2413
 prec_73T19_73T20#120:         - _73T20#67 + _73T19#68 >= 43
 LB_73T18#121:                 _73T18#69 >= 2620
 prec_73T18_73T19#122:         - _73T19#68 + _73T18#69 >= 207
 LB_73T17#123:                 _73T17#70 >= 2930
 prec_73T17_73T18#124:         - _73T18#69 + _73T17#70 >= 310
 LB_73T16#125:                 _73T16#71 >= 3170
 prec_73T16_73T17#126:         - _73T17#70 + _73T16#71 >= 240
 LB_73T15#127:                 _73T15#72 >= 3350
 prec_73T15_73T16#128:         - _73T16#71 + _73T15#72 >= 180
 LB_73T14#129:                 _73T14#73 >= 3920
 prec_73T14_73T15#130:         - _73T15#72 + _73T14#73 >= 570
 LB_73T13#131:                 _73T13#74 >= 4370
 prec_73T13_73T14#132:         - _73T14#73 + _73T13#74 >= 450
 LB_73T12#133:                 _73T12#75 >= 4484
 prec_73T12_73T13#134:         - _73T13#74 + _73T12#75 >= 114
 LB_73T11#135:                 _73T11#76 >= 4670
 prec_73T11_73T12#136:         - _73T12#75 + _73T11#76 >= 186
 LB_73T10#137:                 _73T10#77 >= 4998
 prec_73T10_73T11#138:         - _73T11#76 + _73T10#77 >= 328
 LB_73T9#139:                  _73T9#78 >= 5203
 prec_73T9_73T10#140:          - _73T10#77 + _73T9#78 >= 205
 LB_73T8#141:                  _73T8#79 >= 5510
 prec_73T8_73T9#142:           - _73T9#78 + _73T8#79 >= 307
 LB_73T7#143:                  _73T7#80 >= 5785
 prec_73T7_73T8#144:           - _73T8#79 + _73T7#80 >= 275
 LB_73T6#145:                  _73T6#81 >= 6320
 prec_73T6_73T7#146:           - _73T7#80 + _73T6#81 >= 535
 _73step1#147:                 _73T6#81 - 75318 _73step1#82 <= 6320
 _73step2#148:                 _73T6#81 - 75138 _73step2#83 <= 6500
 _73step3#149:                 _73T6#81 - 74958 _73step3#84 <= 6680
 LB_43T11#150:                 _43T11#85 >= 4468
 LB_43T12#151:                 _43T12#86 >= 4808
 prec_43T12_43T11#152:         - _43T11#85 + _43T12#86 >= 340
 LB_43T13#153:                 _43T13#87 >= 4921
 prec_43T13_43T12#154:         - _43T12#86 + _43T13#87 >= 113
 LB_43T14#155:                 _43T14#88 >= 4996
 prec_43T14_43T13#156:         - _43T13#87 + _43T14#88 >= 75
 LB_43T15#157:                 _43T15#89 >= 5814
 prec_43T15_43T14#158:         - _43T14#88 + _43T15#89 >= 818
 LB_43T16#159:                 _43T16#90 >= 5918
 prec_43T16_43T15#160:         - _43T15#89 + _43T16#90 >= 104
 LB_43T17#161:                 _43T17#91 >= 6235
 prec_43T17_43T16#162:         - _43T16#90 + _43T17#91 >= 317
 LB_43T18#163:                 _43T18#92 >= 6380
 prec_43T18_43T17#164:         - _43T17#91 + _43T18#92 >= 145
 LB_43T19#165:                 _43T19#93 >= 6557
 prec_43T19_43T18#166:         - _43T18#92 + _43T19#93 >= 177
 LB_43T20#167:                 _43T20#94 >= 6687
 prec_43T20_43T19#168:         - _43T19#93 + _43T20#94 >= 130
 LB_43T21#169:                 _43T21#95 >= 6730
 prec_43T21_43T20#170:         - _43T20#94 + _43T21#95 >= 43
 LB_43T22#171:                 _43T22#96 >= 6874
 prec_43T22_43T21#172:         - _43T21#95 + _43T22#96 >= 144
 LB_43T23#173:                 _43T23#97 >= 7435
 prec_43T23_43T22#174:         - _43T22#96 + _43T23#97 >= 561
 LB_43T24#175:                 _43T24#98 >= 7518
 prec_43T24_43T23#176:         - _43T23#97 + _43T24#98 >= 83
 LB_43T25#177:                 _43T25#99 >= 7657
 prec_43T25_43T24#178:         - _43T24#98 + _43T25#99 >= 139
 _43step1#179:                 _43T25#99 - 75318 _43step1#100 <= 3189
 _43step2#180:                 _43T25#99 - 75138 _43step2#101 <= 3369
 _43step3#181:                 _43T25#99 - 74958 _43step3#102 <= 3549
 LB_63T5#182:                  _63T5#103 >= 4620
 LB_63T4#183:                  _63T4#104 >= 6852
 prec_63T4_63T5#184:           - _63T5#103 + _63T4#104 >= 2232
 LB_63T3#185:                  _63T3#105 >= 7160
 prec_63T3_63T4#186:           - _63T4#104 + _63T3#105 >= 308
 LB_63T2#187:                  _63T2#106 >= 8220
 prec_63T2_63T3#188:           - _63T3#105 + _63T2#106 >= 1060
 LB_63T1#189:                  _63T1#107 >= 8769
 prec_63T1_63T2#190:           - _63T2#106 + _63T1#107 >= 549
 _63step1#191:                 _63T1#107 - 75318 _63step1#108 <= 8769
 _63step2#192:                 _63T1#107 - 75138 _63step2#109 <= 8949
 _63step3#193:                 _63T1#107 - 74958 _63step3#110 <= 9129
 Domain_63T1_vs_64T1#194:      _63T1prec64T1#111 >= 0
 Domain_63T2_vs_64T2#195:      _63T2prec64T2#112 >= 0
 Domain_63T3_vs_64T3#196:      _63T3prec64T3#113 >= 0
 Domain_63T4_vs_64T4#197:      _63T4prec64T4#114 >= 0
 InitiDisj_63T5_vs_64T5#198:   _64T5#11 - _63T5#103 + 75318 _63T5prec64T5#115
                               >= 2232
 InitiDisj_64T5_vs_63T5#199:   - _64T5#11 + _63T5#103 - 75318 _63T5prec64T5#115
                               >= -74705
 Domain_56T6_vs_64T6#200:      _56T6prec64T6#116 >= 0
 Domain_56T6_vs_65T6#201:      _56T6prec65T6#117 >= 0
 Domain_56T6_vs_73T6#202:      _56T6prec73T6#118 >= 0
 Domain_64T6_vs_65T6#203:      _64T6prec65T6#119 >= 0
 Domain_64T6_vs_73T6#204:      _64T6prec73T6#120 >= 0
 Domain_65T6_vs_73T6#205:      _65T6prec73T6#121 >= 0
 Domain_56T7_vs_64T7#206:      _56T7prec64T7#122 >= 0
 Domain_56T7_vs_65T7#207:      _56T7prec65T7#123 >= 0
 Domain_56T7_vs_73T7#208:      _56T7prec73T7#124 >= 0
 Domain_64T7_vs_65T7#209:      _64T7prec65T7#125 >= 0
 Domain_64T7_vs_73T7#210:      _64T7prec73T7#126 >= 0
 Domain_65T7_vs_73T7#211:      _65T7prec73T7#127 >= 0
 Domain_56T8_vs_64T8#212:      _56T8prec64T8#128 >= 0
 Domain_56T8_vs_65T8#213:      _56T8prec65T8#129 >= 0
 Domain_56T8_vs_73T8#214:      _56T8prec73T8#130 >= 0
 Domain_64T8_vs_65T8#215:      _64T8prec65T8#131 >= 0
 Domain_64T8_vs_73T8#216:      _64T8prec73T8#132 >= 0
 Domain_65T8_vs_73T8#217:      _65T8prec73T8#133 >= 0
 Domain_56T9_vs_64T9#218:      _56T9prec64T9#134 >= 0
 Domain_56T9_vs_65T9#219:      _56T9prec65T9#135 >= 0
 Domain_56T9_vs_73T9#220:      _56T9prec73T9#136 >= 0
 Domain_64T9_vs_65T9#221:      _64T9prec65T9#137 >= 0
 Domain_64T9_vs_73T9#222:      _64T9prec73T9#138 >= 0
 Domain_65T9_vs_73T9#223:      _65T9prec73T9#139 >= 0
 Domain_56T10_vs_64T10#224:    _56T10prec64T10#140 >= 0
 Domain_56T10_vs_65T10#225:    _56T10prec65T10#141 >= 0
 Domain_56T10_vs_73T10#226:    _56T10prec73T10#142 >= 0
 Domain_64T10_vs_65T10#227:    _64T10prec65T10#143 >= 0
 Domain_64T10_vs_73T10#228:    _64T10prec73T10#144 >= 0
 Domain_65T10_vs_73T10#229:    _65T10prec73T10#145 >= 0
 Domain_43T11_vs_56T11#230:    _43T11prec56T11#146 >= 0
 Domain_43T11_vs_64T11#231:    _43T11prec64T11#147 >= 0
 Domain_43T11_vs_65T11#232:    _43T11prec65T11#148 >= 0
 InitiDisj_43T11_vs_73T11#233: _73T11#76 - _43T11#85
                               + 75318 _43T11prec73T11#149 >= 340
 InitiDisj_73T11_vs_43T11#234: - _73T11#76 + _43T11#85
                               - 75318 _43T11prec73T11#149 >= -74990
 Domain_56T11_vs_64T11#235:    _56T11prec64T11#150 >= 0
 Domain_56T11_vs_65T11#236:    _56T11prec65T11#151 >= 0
 Domain_56T11_vs_73T11#237:    _56T11prec73T11#152 >= 0
 Domain_64T11_vs_65T11#238:    _64T11prec65T11#153 >= 0
 Domain_64T11_vs_73T11#239:    _64T11prec73T11#154 >= 0
 Domain_65T11_vs_73T11#240:    _65T11prec73T11#155 >= 0
 Domain_43T12_vs_64T12#241:    _43T12prec64T12#156 >= 0
 Domain_43T12_vs_65T12#242:    _43T12prec65T12#157 >= 0
 Domain_43T12_vs_73T12#243:    _43T12prec73T12#158 >= 0
 Domain_64T12_vs_65T12#244:    _64T12prec65T12#159 >= 0
 Domain_64T12_vs_73T12#245:    _64T12prec73T12#160 >= 0
 Domain_65T12_vs_73T12#246:    _65T12prec73T12#161 >= 0
 Domain_43T13_vs_64T13#247:    _43T13prec64T13#162 >= 0
 Domain_43T13_vs_65T13#248:    _43T13prec65T13#163 >= 0
 Domain_43T13_vs_73T13#249:    _43T13prec73T13#164 >= 0
 Domain_64T13_vs_65T13#250:    _64T13prec65T13#165 >= 0
 Domain_64T13_vs_73T13#251:    _64T13prec73T13#166 >= 0
 Domain_65T13_vs_73T13#252:    _65T13prec73T13#167 >= 0
 Domain_43T14_vs_64T14#253:    _43T14prec64T14#168 >= 0
 Domain_43T14_vs_65T14#254:    _43T14prec65T14#169 >= 0
 Domain_43T14_vs_73T14#255:    _43T14prec73T14#170 >= 0
 Domain_64T14_vs_65T14#256:    _64T14prec65T14#171 >= 0
 Domain_64T14_vs_73T14#257:    _64T14prec73T14#172 >= 0
 Domain_65T14_vs_73T14#258:    _65T14prec73T14#173 >= 0
 Domain_43T15_vs_64T15#259:    _43T15prec64T15#174 >= 0
 Domain_43T15_vs_65T15#260:    _43T15prec65T15#175 >= 0
 Domain_43T15_vs_73T15#261:    _43T15prec73T15#176 >= 0
 Domain_64T15_vs_65T15#262:    _64T15prec65T15#177 >= 0
 Domain_64T15_vs_73T15#263:    _64T15prec73T15#178 >= 0
 Domain_65T15_vs_73T15#264:    _65T15prec73T15#179 >= 0
 Domain_43T16_vs_64T16#265:    _43T16prec64T16#180 >= 0
 Domain_43T16_vs_65T16#266:    _43T16prec65T16#181 >= 0
 Domain_43T16_vs_73T16#267:    _43T16prec73T16#182 >= 0
 Domain_64T16_vs_65T16#268:    _64T16prec65T16#183 >= 0
 Domain_64T16_vs_73T16#269:    _64T16prec73T16#184 >= 0
 Domain_65T16_vs_73T16#270:    _65T16prec73T16#185 >= 0
 Domain_43T17_vs_65T17#271:    _43T17prec65T17#186 >= 0
 Domain_43T17_vs_73T17#272:    _43T17prec73T17#187 >= 0
 Domain_65T17_vs_73T17#273:    _65T17prec73T17#188 >= 0
 Domain_43T18_vs_65T18#274:    _43T18prec65T18#189 >= 0
 Domain_43T18_vs_66T18#275:    _43T18prec66T18#190 >= 0
 Domain_43T18_vs_73T18#276:    _43T18prec73T18#191 >= 0
 Domain_65T18_vs_66T18#277:    _65T18prec66T18#192 >= 0
 Domain_65T18_vs_73T18#278:    _65T18prec73T18#193 >= 0
 Domain_66T18_vs_73T18#279:    _66T18prec73T18#194 >= 0
 Domain_43T19_vs_65T19#280:    _43T19prec65T19#195 >= 0
 Domain_43T19_vs_66T19#281:    _43T19prec66T19#196 >= 0
 Domain_43T19_vs_73T19#282:    _43T19prec73T19#197 >= 0
 Domain_65T19_vs_66T19#283:    _65T19prec66T19#198 >= 0
 Domain_65T19_vs_73T19#284:    _65T19prec73T19#199 >= 0
 Domain_66T19_vs_73T19#285:    _66T19prec73T19#200 >= 0
 Domain_43T20_vs_65T20#286:    _43T20prec65T20#201 >= 0
 Domain_43T20_vs_66T20#287:    _43T20prec66T20#202 >= 0
 Domain_43T20_vs_73T20#288:    _43T20prec73T20#203 >= 0
 Domain_65T20_vs_66T20#289:    _65T20prec66T20#204 >= 0
 Domain_65T20_vs_73T20#290:    _65T20prec73T20#205 >= 0
 Domain_66T20_vs_73T20#291:    _66T20prec73T20#206 >= 0
 Domain_43T21_vs_65T21#292:    _43T21prec65T21#207 >= 0
 Domain_43T21_vs_66T21#293:    _43T21prec66T21#208 >= 0
 Domain_43T21_vs_73T21#294:    _43T21prec73T21#209 >= 0
 Domain_65T21_vs_66T21#295:    _65T21prec66T21#210 >= 0
 Domain_65T21_vs_73T21#296:    _65T21prec73T21#211 >= 0
 Domain_66T21_vs_73T21#297:    _66T21prec73T21#212 >= 0
 Domain_43T22_vs_65T22#298:    _43T22prec65T22#213 >= 0
 Domain_43T22_vs_66T22#299:    _43T22prec66T22#214 >= 0
 Domain_43T22_vs_73T22#300:    _43T22prec73T22#215 >= 0
 Domain_65T22_vs_66T22#301:    _65T22prec66T22#216 >= 0
 Domain_65T22_vs_73T22#302:    _65T22prec73T22#217 >= 0
 Domain_66T22_vs_73T22#303:    _66T22prec73T22#218 >= 0
 Domain_43T23_vs_65T23#304:    _43T23prec65T23#219 >= 0
 Domain_43T23_vs_66T23#305:    _43T23prec66T23#220 >= 0
 Domain_43T23_vs_73T23#306:    _43T23prec73T23#221 >= 0
 Domain_65T23_vs_66T23#307:    _65T23prec66T23#222 >= 0
 Domain_65T23_vs_73T23#308:    _65T23prec73T23#223 >= 0
 Domain_66T23_vs_73T23#309:    _66T23prec73T23#224 >= 0
 Domain_43T24_vs_65T24#310:    _43T24prec65T24#225 >= 0
 Domain_43T24_vs_66T24#311:    _43T24prec66T24#226 >= 0
 Domain_43T24_vs_73T24#312:    _43T24prec73T24#227 >= 0
 Domain_65T24_vs_66T24#313:    _65T24prec66T24#228 >= 0
 Domain_65T24_vs_73T24#314:    _65T24prec73T24#229 >= 0
 Domain_66T24_vs_73T24#315:    _66T24prec73T24#230 >= 0
 Domain_43T25_vs_65T25#316:    _43T25prec65T25#231 >= 0
 Domain_43T25_vs_66T25#317:    _43T25prec66T25#232 >= 0
 Domain_43T25_vs_73T25#318:    _43T25prec73T25#233 >= 0
 Domain_65T25_vs_66T25#319:    _65T25prec66T25#234 >= 0
 Domain_65T25_vs_73T25#320:    _65T25prec73T25#235 >= 0
 InitiDisj_66T25_vs_73T25#321: - _66T25#49 + _73T25#62
                               + 75318 _66T25prec73T25#236 >= 190
 InitiDisj_73T25_vs_66T25#322: _66T25#49 - _73T25#62
                               - 75318 _66T25prec73T25#236 >= -75018
Bounds
 0 <= _64T16#0 <= 2147483647
 0 <= _64T15#1 <= 2147483647
 0 <= _64T14#2 <= 2147483647
 0 <= _64T13#3 <= 2147483647
 0 <= _64T12#4 <= 2147483647
 0 <= _64T11#5 <= 2147483647
 0 <= _64T10#6 <= 2147483647
 0 <= _64T9#7 <= 2147483647
 0 <= _64T8#8 <= 2147483647
 0 <= _64T7#9 <= 2147483647
 0 <= _64T6#10 <= 2147483647
 0 <= _64T5#11 <= 2147483647
 0 <= _64T4#12 <= 2147483647
 0 <= _64T3#13 <= 2147483647
 0 <= _64T2#14 <= 2147483647
 0 <= _64T1#15 <= 2147483647
 0 <= _64step1#16 <= 1
 0 <= _64step2#17 <= 1
 0 <= _64step3#18 <= 1
 0 <= _65T25#19 <= 2147483647
 0 <= _65T24#20 <= 2147483647
 0 <= _65T23#21 <= 2147483647
 0 <= _65T22#22 <= 2147483647
 0 <= _65T21#23 <= 2147483647
 0 <= _65T20#24 <= 2147483647
 0 <= _65T19#25 <= 2147483647
 0 <= _65T18#26 <= 2147483647
 0 <= _65T17#27 <= 2147483647
 0 <= _65T16#28 <= 2147483647
 0 <= _65T15#29 <= 2147483647
 0 <= _65T14#30 <= 2147483647
 0 <= _65T13#31 <= 2147483647
 0 <= _65T12#32 <= 2147483647
 0 <= _65T11#33 <= 2147483647
 0 <= _65T10#34 <= 2147483647
 0 <= _65T9#35 <= 2147483647
 0 <= _65T8#36 <= 2147483647
 0 <= _65T7#37 <= 2147483647
 0 <= _65T6#38 <= 2147483647
 0 <= _65step1#39 <= 1
 0 <= _65step2#40 <= 1
 0 <= _65step3#41 <= 1
 0 <= _66T18#42 <= 2147483647
 0 <= _66T19#43 <= 2147483647
 0 <= _66T20#44 <= 2147483647
 0 <= _66T21#45 <= 2147483647
 0 <= _66T22#46 <= 2147483647
 0 <= _66T23#47 <= 2147483647
 0 <= _66T24#48 <= 2147483647
 0 <= _66T25#49 <= 2147483647
 0 <= _66step1#50 <= 1
 0 <= _66step2#51 <= 1
 0 <= _66step3#52 <= 1
 0 <= _56T11#53 <= 2147483647
 0 <= _56T10#54 <= 2147483647
 0 <= _56T9#55 <= 2147483647
 0 <= _56T8#56 <= 2147483647
 0 <= _56T7#57 <= 2147483647
 0 <= _56T6#58 <= 2147483647
 0 <= _56step1#59 <= 1
 0 <= _56step2#60 <= 1
 0 <= _56step3#61 <= 1
 0 <= _73T25#62 <= 2147483647
 0 <= _73T24#63 <= 2147483647
 0 <= _73T23#64 <= 2147483647
 0 <= _73T22#65 <= 2147483647
 0 <= _73T21#66 <= 2147483647
 0 <= _73T20#67 <= 2147483647
 0 <= _73T19#68 <= 2147483647
 0 <= _73T18#69 <= 2147483647
 0 <= _73T17#70 <= 2147483647
 0 <= _73T16#71 <= 2147483647
 0 <= _73T15#72 <= 2147483647
 0 <= _73T14#73 <= 2147483647
 0 <= _73T13#74 <= 2147483647
 0 <= _73T12#75 <= 2147483647
 0 <= _73T11#76 <= 2147483647
 0 <= _73T10#77 <= 2147483647
 0 <= _73T9#78 <= 2147483647
 0 <= _73T8#79 <= 2147483647
 0 <= _73T7#80 <= 2147483647
 0 <= _73T6#81 <= 2147483647
 0 <= _73step1#82 <= 1
 0 <= _73step2#83 <= 1
 0 <= _73step3#84 <= 1
 0 <= _43T11#85 <= 2147483647
 0 <= _43T12#86 <= 2147483647
 0 <= _43T13#87 <= 2147483647
 0 <= _43T14#88 <= 2147483647
 0 <= _43T15#89 <= 2147483647
 0 <= _43T16#90 <= 2147483647
 0 <= _43T17#91 <= 2147483647
 0 <= _43T18#92 <= 2147483647
 0 <= _43T19#93 <= 2147483647
 0 <= _43T20#94 <= 2147483647
 0 <= _43T21#95 <= 2147483647
 0 <= _43T22#96 <= 2147483647
 0 <= _43T23#97 <= 2147483647
 0 <= _43T24#98 <= 2147483647
 0 <= _43T25#99 <= 2147483647
 0 <= _43step1#100 <= 1
 0 <= _43step2#101 <= 1
 0 <= _43step3#102 <= 1
 0 <= _63T5#103 <= 2147483647
 0 <= _63T4#104 <= 2147483647
 0 <= _63T3#105 <= 2147483647
 0 <= _63T2#106 <= 2147483647
 0 <= _63T1#107 <= 2147483647
 0 <= _63step1#108 <= 1
 0 <= _63step2#109 <= 1
 0 <= _63step3#110 <= 1
 0 <= _63T1prec64T1#111 <= 1
 0 <= _63T2prec64T2#112 <= 1
 0 <= _63T3prec64T3#113 <= 1
 0 <= _63T4prec64T4#114 <= 1
 0 <= _63T5prec64T5#115 <= 1
 0 <= _56T6prec64T6#116 <= 1
 0 <= _56T6prec65T6#117 <= 1
 0 <= _56T6prec73T6#118 <= 1
 0 <= _64T6prec65T6#119 <= 1
 0 <= _64T6prec73T6#120 <= 1
 0 <= _65T6prec73T6#121 <= 1
 0 <= _56T7prec64T7#122 <= 1
 0 <= _56T7prec65T7#123 <= 1
 0 <= _56T7prec73T7#124 <= 1
 0 <= _64T7prec65T7#125 <= 1
 0 <= _64T7prec73T7#126 <= 1
 0 <= _65T7prec73T7#127 <= 1
 0 <= _56T8prec64T8#128 <= 1
 0 <= _56T8prec65T8#129 <= 1
 0 <= _56T8prec73T8#130 <= 1
 0 <= _64T8prec65T8#131 <= 1
 0 <= _64T8prec73T8#132 <= 1
 0 <= _65T8prec73T8#133 <= 1
 0 <= _56T9prec64T9#134 <= 1
 0 <= _56T9prec65T9#135 <= 1
 0 <= _56T9prec73T9#136 <= 1
 0 <= _64T9prec65T9#137 <= 1
 0 <= _64T9prec73T9#138 <= 1
 0 <= _65T9prec73T9#139 <= 1
 0 <= _56T10prec64T10#140 <= 1
 0 <= _56T10prec65T10#141 <= 1
 0 <= _56T10prec73T10#142 <= 1
 0 <= _64T10prec65T10#143 <= 1
 0 <= _64T10prec73T10#144 <= 1
 0 <= _65T10prec73T10#145 <= 1
 0 <= _43T11prec56T11#146 <= 1
 0 <= _43T11prec64T11#147 <= 1
 0 <= _43T11prec65T11#148 <= 1
 0 <= _43T11prec73T11#149 <= 1
 0 <= _56T11prec64T11#150 <= 1
 0 <= _56T11prec65T11#151 <= 1
 0 <= _56T11prec73T11#152 <= 1
 0 <= _64T11prec65T11#153 <= 1
 0 <= _64T11prec73T11#154 <= 1
 0 <= _65T11prec73T11#155 <= 1
 0 <= _43T12prec64T12#156 <= 1
 0 <= _43T12prec65T12#157 <= 1
 0 <= _43T12prec73T12#158 <= 1
 0 <= _64T12prec65T12#159 <= 1
 0 <= _64T12prec73T12#160 <= 1
 0 <= _65T12prec73T12#161 <= 1
 0 <= _43T13prec64T13#162 <= 1
 0 <= _43T13prec65T13#163 <= 1
 0 <= _43T13prec73T13#164 <= 1
 0 <= _64T13prec65T13#165 <= 1
 0 <= _64T13prec73T13#166 <= 1
 0 <= _65T13prec73T13#167 <= 1
 0 <= _43T14prec64T14#168 <= 1
 0 <= _43T14prec65T14#169 <= 1
 0 <= _43T14prec73T14#170 <= 1
 0 <= _64T14prec65T14#171 <= 1
 0 <= _64T14prec73T14#172 <= 1
 0 <= _65T14prec73T14#173 <= 1
 0 <= _43T15prec64T15#174 <= 1
 0 <= _43T15prec65T15#175 <= 1
 0 <= _43T15prec73T15#176 <= 1
 0 <= _64T15prec65T15#177 <= 1
 0 <= _64T15prec73T15#178 <= 1
 0 <= _65T15prec73T15#179 <= 1
 0 <= _43T16prec64T16#180 <= 1
 0 <= _43T16prec65T16#181 <= 1
 0 <= _43T16prec73T16#182 <= 1
 0 <= _64T16prec65T16#183 <= 1
 0 <= _64T16prec73T16#184 <= 1
 0 <= _65T16prec73T16#185 <= 1
 0 <= _43T17prec65T17#186 <= 1
 0 <= _43T17prec73T17#187 <= 1
 0 <= _65T17prec73T17#188 <= 1
 0 <= _43T18prec65T18#189 <= 1
 0 <= _43T18prec66T18#190 <= 1
 0 <= _43T18prec73T18#191 <= 1
 0 <= _65T18prec66T18#192 <= 1
 0 <= _65T18prec73T18#193 <= 1
 0 <= _66T18prec73T18#194 <= 1
 0 <= _43T19prec65T19#195 <= 1
 0 <= _43T19prec66T19#196 <= 1
 0 <= _43T19prec73T19#197 <= 1
 0 <= _65T19prec66T19#198 <= 1
 0 <= _65T19prec73T19#199 <= 1
 0 <= _66T19prec73T19#200 <= 1
 0 <= _43T20prec65T20#201 <= 1
 0 <= _43T20prec66T20#202 <= 1
 0 <= _43T20prec73T20#203 <= 1
 0 <= _65T20prec66T20#204 <= 1
 0 <= _65T20prec73T20#205 <= 1
 0 <= _66T20prec73T20#206 <= 1
 0 <= _43T21prec65T21#207 <= 1
 0 <= _43T21prec66T21#208 <= 1
 0 <= _43T21prec73T21#209 <= 1
 0 <= _65T21prec66T21#210 <= 1
 0 <= _65T21prec73T21#211 <= 1
 0 <= _66T21prec73T21#212 <= 1
 0 <= _43T22prec65T22#213 <= 1
 0 <= _43T22prec66T22#214 <= 1
 0 <= _43T22prec73T22#215 <= 1
 0 <= _65T22prec66T22#216 <= 1
 0 <= _65T22prec73T22#217 <= 1
 0 <= _66T22prec73T22#218 <= 1
 0 <= _43T23prec65T23#219 <= 1
 0 <= _43T23prec66T23#220 <= 1
 0 <= _43T23prec73T23#221 <= 1
 0 <= _65T23prec66T23#222 <= 1
 0 <= _65T23prec73T23#223 <= 1
 0 <= _66T23prec73T23#224 <= 1
 0 <= _43T24prec65T24#225 <= 1
 0 <= _43T24prec66T24#226 <= 1
 0 <= _43T24prec73T24#227 <= 1
 0 <= _65T24prec66T24#228 <= 1
 0 <= _65T24prec73T24#229 <= 1
 0 <= _66T24prec73T24#230 <= 1
 0 <= _43T25prec65T25#231 <= 1
 0 <= _43T25prec66T25#232 <= 1
 0 <= _43T25prec73T25#233 <= 1
 0 <= _65T25prec66T25#234 <= 1
 0 <= _65T25prec73T25#235 <= 1
 0 <= _66T25prec73T25#236 <= 1
Binaries
 _64step1#16  _64step2#17  _64step3#18  _65step1#39  _65step2#40  _65step3#41 
 _66step1#50  _66step2#51  _66step3#52  _56step1#59  _56step2#60  _56step3#61 
 _73step1#82  _73step2#83  _73step3#84  _43step1#100  _43step2#101 
 _43step3#102  _63step1#108  _63step2#109  _63step3#110  _63T1prec64T1#111 
 _63T2prec64T2#112  _63T3prec64T3#113  _63T4prec64T4#114  _63T5prec64T5#115 
 _56T6prec64T6#116  _56T6prec65T6#117  _56T6prec73T6#118  _64T6prec65T6#119 
 _64T6prec73T6#120  _65T6prec73T6#121  _56T7prec64T7#122  _56T7prec65T7#123 
 _56T7prec73T7#124  _64T7prec65T7#125  _64T7prec73T7#126  _65T7prec73T7#127 
 _56T8prec64T8#128  _56T8prec65T8#129  _56T8prec73T8#130  _64T8prec65T8#131 
 _64T8prec73T8#132  _65T8prec73T8#133  _56T9prec64T9#134  _56T9prec65T9#135 
 _56T9prec73T9#136  _64T9prec65T9#137  _64T9prec73T9#138  _65T9prec73T9#139 
 _56T10prec64T10#140  _56T10prec65T10#141  _56T10prec73T10#142 
 _64T10prec65T10#143  _64T10prec73T10#144  _65T10prec73T10#145 
 _43T11prec56T11#146  _43T11prec64T11#147  _43T11prec65T11#148 
 _43T11prec73T11#149  _56T11prec64T11#150  _56T11prec65T11#151 
 _56T11prec73T11#152  _64T11prec65T11#153  _64T11prec73T11#154 
 _65T11prec73T11#155  _43T12prec64T12#156  _43T12prec65T12#157 
 _43T12prec73T12#158  _64T12prec65T12#159  _64T12prec73T12#160 
 _65T12prec73T12#161  _43T13prec64T13#162  _43T13prec65T13#163 
 _43T13prec73T13#164  _64T13prec65T13#165  _64T13prec73T13#166 
 _65T13prec73T13#167  _43T14prec64T14#168  _43T14prec65T14#169 
 _43T14prec73T14#170  _64T14prec65T14#171  _64T14prec73T14#172 
 _65T14prec73T14#173  _43T15prec64T15#174  _43T15prec65T15#175 
 _43T15prec73T15#176  _64T15prec65T15#177  _64T15prec73T15#178 
 _65T15prec73T15#179  _43T16prec64T16#180  _43T16prec65T16#181 
 _43T16prec73T16#182  _64T16prec65T16#183  _64T16prec73T16#184 
 _65T16prec73T16#185  _43T17prec65T17#186  _43T17prec73T17#187 
 _65T17prec73T17#188  _43T18prec65T18#189  _43T18prec66T18#190 
 _43T18prec73T18#191  _65T18prec66T18#192  _65T18prec73T18#193 
 _66T18prec73T18#194  _43T19prec65T19#195  _43T19prec66T19#196 
 _43T19prec73T19#197  _65T19prec66T19#198  _65T19prec73T19#199 
 _66T19prec73T19#200  _43T20prec65T20#201  _43T20prec66T20#202 
 _43T20prec73T20#203  _65T20prec66T20#204  _65T20prec73T20#205 
 _66T20prec73T20#206  _43T21prec65T21#207  _43T21prec66T21#208 
 _43T21prec73T21#209  _65T21prec66T21#210  _65T21prec73T21#211 
 _66T21prec73T21#212  _43T22prec65T22#213  _43T22prec66T22#214 
 _43T22prec73T22#215  _65T22prec66T22#216  _65T22prec73T22#217 
 _66T22prec73T22#218  _43T23prec65T23#219  _43T23prec66T23#220 
 _43T23prec73T23#221  _65T23prec66T23#222  _65T23prec73T23#223 
 _66T23prec73T23#224  _43T24prec65T24#225  _43T24prec66T24#226 
 _43T24prec73T24#227  _65T24prec66T24#228  _65T24prec73T24#229 
 _66T24prec73T24#230  _43T25prec65T25#231  _43T25prec66T25#232 
 _43T25prec73T25#233  _65T25prec66T25#234  _65T25prec73T25#235 
 _66T25prec73T25#236 
End
