"""
    Validation playground.

run 0:
    predictors = ['app', 'channel',  'device', 'ip',  'os',
       'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
       'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
       'count_ip_app_channel_hour', ]    
    
[10]    train's auc: 0.959527   h4's auc: 0.950359      h5's auc: 0.955799      h9's auc: 0.950592      h10's auc: 0.951147     h13's auc: 0.959001     h14's auc: 0.965224
[20]    train's auc: 0.965493   h4's auc: 0.95573       h5's auc: 0.960708      h9's auc: 0.956189      h10's auc: 0.955926     h13's auc: 0.963335     h14's auc: 0.969723
[30]    train's auc: 0.968663   h4's auc: 0.959477      h5's auc: 0.963473      h9's auc: 0.95933       h10's auc: 0.958812     h13's auc: 0.965431     h14's auc: 0.971698
[40]    train's auc: 0.970679   h4's auc: 0.960195      h5's auc: 0.964605      h9's auc: 0.961153      h10's auc: 0.960396     h13's auc: 0.967443     h14's auc: 0.97354
[50]    train's auc: 0.972083   h4's auc: 0.961635      h5's auc: 0.965603      h9's auc: 0.962678      h10's auc: 0.962149     h13's auc: 0.969177     h14's auc: 0.975069    

run 1:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device'
    ]
[10]    train's auc: 0.960376   h4's auc: 0.951176      h5's auc: 0.955174      h9's auc: 0.951835      h10's auc: 0.952416     h13's auc: 0.959729     h14's auc: 0.966331
[20]    train's auc: 0.968236   h4's auc: 0.95843       h5's auc: 0.962334      h9's auc: 0.958911      h10's auc: 0.95786      h13's auc: 0.965024     h14's auc: 0.971281
[30]    train's auc: 0.968936   h4's auc: 0.958543      h5's auc: 0.962655      h9's auc: 0.959729      h10's auc: 0.958554     h13's auc: 0.96558      h14's auc: 0.971882
[40]    train's auc: 0.970379   h4's auc: 0.959628      h5's auc: 0.963854      h9's auc: 0.96125       h10's auc: 0.9604       h13's auc: 0.967577     h14's auc: 0.973552
[50]    train's auc: 0.972173   h4's auc: 0.961355      h5's auc: 0.96519       h9's auc: 0.962771      h10's auc: 0.962334     h13's auc: 0.969326     h14's auc: 0.975364


run 2:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'mtbc_ip_device'
    ]    
[10]    train's auc: 0.960376   h4's auc: 0.951176      h5's auc: 0.955174      h9's auc: 0.951835      h10's auc: 0.952416     h13's auc: 0.959729     h14's auc: 0.966331
[20]    train's auc: 0.968236   h4's auc: 0.95843       h5's auc: 0.962334      h9's auc: 0.958911      h10's auc: 0.95786      h13's auc: 0.965024     h14's auc: 0.971281
[30]    train's auc: 0.968982   h4's auc: 0.958638      h5's auc: 0.962773      h9's auc: 0.959829      h10's auc: 0.958552     h13's auc: 0.965612     h14's auc: 0.971903
[40]    train's auc: 0.970509   h4's auc: 0.959796      h5's auc: 0.964033      h9's auc: 0.961324      h10's auc: 0.960471     h13's auc: 0.96752      h14's auc: 0.973588
[50]    train's auc: 0.972075   h4's auc: 0.96123       h5's auc: 0.965089      h9's auc: 0.962646      h10's auc: 0.962022     h13's auc: 0.969182     h14's auc: 0.975126    


run3:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device',
        'mtbc_ip_device',
    ]
[10]    train's auc: 0.958102   h4's auc: 0.946473      h5's auc: 0.950017      h9's auc: 0.949173      h10's auc: 0.950113     h13's auc: 0.958214     h14's auc: 0.965805
[20]    train's auc: 0.967429   h4's auc: 0.955952      h5's auc: 0.960304      h9's auc: 0.95774       h10's auc: 0.956678     h13's auc: 0.964073     h14's auc: 0.970948
[30]    train's auc: 0.968735   h4's auc: 0.958143      h5's auc: 0.96206       h9's auc: 0.959387      h10's auc: 0.958446     h13's auc: 0.965206     h14's auc: 0.971613
[40]    train's auc: 0.970618   h4's auc: 0.959987      h5's auc: 0.964039      h9's auc: 0.961314      h10's auc: 0.96054      h13's auc: 0.967791     h14's auc: 0.973817
[50]    train's auc: 0.972044   h4's auc: 0.96148       h5's auc: 0.965335      h9's auc: 0.962771      h10's auc: 0.962435     h13's auc: 0.96967      h14's auc: 0.975293    

run4:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device', 'delta-ip-device-os-channel',
        'mtbc_ip_device', 'mtbc_ip_device_os_channel',
    ]

[10]    train's auc: 0.959531   h4's auc: 0.948324      h5's auc: 0.953005      h9's auc: 0.94927       h10's auc: 0.949501     h13's auc: 0.957499 h14's auc: 0.96379
[20]    train's auc: 0.966785   h4's auc: 0.955171      h5's auc: 0.959757      h9's auc: 0.956588      h10's auc: 0.955985     h13's auc: 0.963648 h14's auc: 0.970011
[30]    train's auc: 0.968445   h4's auc: 0.95791       h5's auc: 0.962198      h9's auc: 0.959082      h10's auc: 0.958242     h13's auc: 0.965146     h14's auc: 0.971263
[40]    train's auc: 0.970374   h4's auc: 0.959768      h5's auc: 0.963854      h9's auc: 0.960935      h10's auc: 0.960164     h13's auc: 0.967353     h14's auc: 0.973331
[50]    train's auc: 0.97187    h4's auc: 0.960909      h5's auc: 0.964621      h9's auc: 0.962567      h10's auc: 0.961973     h13's auc: 0.969207     h14's auc: 0.975188

run5:
    train using day==8
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device'
    ]
    
[10]    train's auc: 0.965039   h4's auc: 0.953124      h5's auc: 0.956573      h9's auc: 0.95295       h10's auc: 0.953216     h13's auc: 0.961614     h14's auc: 0.967944
[20]    train's auc: 0.969389   h4's auc: 0.957767      h5's auc: 0.960935      h9's auc: 0.957844      h10's auc: 0.957219     h13's auc: 0.964847     h14's auc: 0.971389
[30]    train's auc: 0.970275   h4's auc: 0.958696      h5's auc: 0.961798      h9's auc: 0.959265      h10's auc: 0.95842      h13's auc: 0.965469     h14's auc: 0.971681
[40]    train's auc: 0.97198    h4's auc: 0.959727      h5's auc: 0.963475      h9's auc: 0.960922      h10's auc: 0.960104     h13's auc: 0.966989     h14's auc: 0.97309
[50]    train's auc: 0.97329    h4's auc: 0.96131       h5's auc: 0.964817      h9's auc: 0.962336      h10's auc: 0.961721     h13's auc: 0.968654     h14's auc: 0.974646

run6:
    train using day==8
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        #'count_ip_app_channel_hour', 
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        #'delta_ip_device'
        'lhood_ip_day_in_test_hh',
        'lhood_app_channel_day_in_test_hh',
        'lhood_ip_hour',
        'lhood_ip_device_hour',        
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
    ]
[10]    train's auc: 0.970903   h4's auc: 0.959015      h5's auc: 0.964386      h9's auc: 0.961648      h10's auc: 0.961056     h13's auc: 0.966639     h14's auc: 0.972521
[20]    train's auc: 0.974574   h4's auc: 0.962869      h5's auc: 0.967562      h9's auc: 0.965182      h10's auc: 0.963935     h13's auc: 0.969606     h14's auc: 0.974935
[30]    train's auc: 0.975921   h4's auc: 0.964595      h5's auc: 0.96921       h9's auc: 0.966697      h10's auc: 0.96558      h13's auc: 0.970307     h14's auc: 0.975602
[40]    train's auc: 0.977953   h4's auc: 0.967292      h5's auc: 0.971751      h9's auc: 0.968905      h10's auc: 0.968073     h13's auc: 0.972673     h14's auc: 0.977673
[50]    train's auc: 0.979267   h4's auc: 0.967725      h5's auc: 0.972387      h9's auc: 0.97006       h10's auc: 0.969344     h13's auc: 0.973932     h14's auc: 0.978771
Did not meet early stopping. Best iteration is:
[50]    train's auc: 0.979267   h4's auc: 0.967725      h5's auc: 0.972387      h9's auc: 0.97006       h10's auc: 0.969344     h13's auc: 0.973932     h14's auc: 0.978771
[('lhood_ip_day_in_test_hh', 0), ('lhood_app_channel_day_in_test_hh', 0), ('lhood_ip_hour', 0), ('lhood_ip_device_hour', 1), ('hour', 6), ('count_ip_os_hour', 8), ('device', 9), ('count_ip_day_hour', 9), ('count_ip_device_hour', 9), ('count_ip_day_app_in_test_hh', 10), ('count_ip_day_device_in_test_hh', 14), ('count_ip_app_hour', 30), ('os', 33), ('count_ip_day_in_test_hh', 36), ('lhood_ip_app_hour', 38), ('lhood_ip_channel_hour', 39), ('channel', 60), ('app', 98)]
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric':'auc',
              'learning_rate': 0.1,
              'num_leaves': 15, 
              'max_depth': -1, 
              #'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
              'max_bin': 250,  
              'subsample': 0.9,  # Subsample ratio of the training instance.
              'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
              'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
              #'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
              #'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
              'nthread': 8,
              'verbose': 0,
              'scale_pos_weight':99.7, # because training data is extremely unbalanced 
    
run7:
    train using day==8
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        #'count_ip_app_channel_hour', 
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        #'delta_ip_device'
        #'lhood_ip_day_in_test_hh',
        #'lhood_app_channel_day_in_test_hh',
        #'lhood_ip_hour',
        #'lhood_ip_device_hour',        
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
    ]    
[10]    train's auc: 0.974769   h4's auc: 0.963476      h5's auc: 0.968866      h9's auc: 0.967181      h10's auc: 0.966385     h13's auc: 0.971319     h14's auc: 0.975215
[20]    train's auc: 0.978476   h4's auc: 0.966495      h5's auc: 0.972048      h9's auc: 0.969669      h10's auc: 0.96903      h13's auc: 0.974281     h14's auc: 0.978222
[30]    train's auc: 0.979668   h4's auc: 0.968354      h5's auc: 0.972838      h9's auc: 0.97087       h10's auc: 0.9704       h13's auc: 0.975607     h14's auc: 0.97949
[40]    train's auc: 0.980941   h4's auc: 0.969604      h5's auc: 0.974066      h9's auc: 0.971887      h10's auc: 0.971339     h13's auc: 0.976676     h14's auc: 0.980188
[50]    train's auc: 0.982477   h4's auc: 0.970871      h5's auc: 0.974923      h9's auc: 0.972672      h10's auc: 0.972522     h13's auc: 0.977487     h14's auc: 0.981103
[('count_ip_day_hour', 24), ('device', 30), ('count_ip_os_hour', 34), ('count_ip_day_app_in_test_hh', 45), ('count_ip_day_device_in_test_hh', 56), ('count_ip_device_hour', 58), ('lhood_ip_channel_hour', 76), ('lhood_ip_app_hour', 102), ('hour', 107), ('count_ip_day_in_test_hh', 113), ('os', 164), ('count_ip_app_hour', 180), ('app', 217), ('channel', 294)]
   


run8:
    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric':'auc',
              'learning_rate': 0.1,
              'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
              'max_depth': -1,  # -1 means no limit
              #'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
              'max_bin': 255,  
              'bagging_freq': 0.8,
              'bagging_freq': 1,
              'bagging_seed': seed,
              #'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
              #'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
              #'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
              'nthread': 8,
              'verbose': 0,
              'scale_pos_weight':99.7, # because training data is extremely unbalanced               
             }
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
    ]
[10]    train's auc: 0.974818   h4's auc: 0.963419      h5's auc: 0.96882       h9's auc: 0.967234      h10's auc: 0.966369     h13's auc: 0.971383     h14's auc: 0.975282
[20]    train's auc: 0.978219   h4's auc: 0.967371      h5's auc: 0.97199       h9's auc: 0.969815      h10's auc: 0.969312     h13's auc: 0.974298     h14's auc: 0.978425
[30]    train's auc: 0.979675   h4's auc: 0.968135      h5's auc: 0.973008      h9's auc: 0.97091       h10's auc: 0.970427     h13's auc: 0.975697     h14's auc: 0.979595
[40]    train's auc: 0.981133   h4's auc: 0.969135      h5's auc: 0.974236      h9's auc: 0.972047      h10's auc: 0.971478     h13's auc: 0.976748     h14's auc: 0.980318
[50]    train's auc: 0.982408   h4's auc: 0.970327      h5's auc: 0.974711      h9's auc: 0.972712      h10's auc: 0.972474     h13's auc: 0.977557     h14's auc: 0.981164
[60]    train's auc: 0.983684   h4's auc: 0.971507      h5's auc: 0.975734      h9's auc: 0.973612      h10's auc: 0.973277     h13's auc: 0.978487     h14's auc: 0.981889
[70]    train's auc: 0.984564   h4's auc: 0.972207      h5's auc: 0.976365      h9's auc: 0.974098      h10's auc: 0.97373      h13's auc: 0.978993     h14's auc: 0.982311
[80]    train's auc: 0.985159   h4's auc: 0.972479      h5's auc: 0.976593      h9's auc: 0.974291      h10's auc: 0.974017     h13's auc: 0.979238     h14's auc: 0.982701
[90]    train's auc: 0.985623   h4's auc: 0.972693      h5's auc: 0.9767        h9's auc: 0.974427      h10's auc: 0.974099     h13's auc: 0.979404     h14's auc: 0.982842
[100]   train's auc: 0.986052   h4's auc: 0.972744      h5's auc: 0.976764      h9's auc: 0.974538      h10's auc: 0.974219     h13's auc: 0.979524     h14's auc: 0.982915
[110]   train's auc: 0.986472   h4's auc: 0.972887      h5's auc: 0.976788      h9's auc: 0.974572      h10's auc: 0.974308     h13's auc: 0.979591     h14's auc: 0.98304
[120]   train's auc: 0.986813   h4's auc: 0.97293       h5's auc: 0.976875      h9's auc: 0.974575      h10's auc: 0.974381     h13's auc: 0.979601     h14's auc: 0.983106



run9:
    train using day==8
    note: fixed lhood type from bool to binned uint.
    note2: calculate lhood using folds/day instead  of whole train to reduce leakage
    
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        #'count_ip_app_channel_hour', 
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        'count_app_day_in_test_hh',
        'count_channel_day_device_in_test_hh',
        #'delta_ip_device'
        #'lhood_ip_day_in_test_hh',
        #'lhood_app_channel_day_in_test_hh',
        #'lhood_ip_hour',
        #'lhood_ip_device_hour',        
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
        #'lhood_channel_hour',
        'lhood_app_hour',
        #'lhood_app_channel_hour',
        #'lhood_os_hour'
        'lhood_channel_hour',
    ]    
[10]    train's auc: 0.968623   h4's auc: 0.956644      h5's auc: 0.960158      h9's auc: 0.958614      h10's auc: 0.957954     h13's auc: 0.965531     h14's auc: 0.971956
[20]    train's auc: 0.972371   h4's auc: 0.96104       h5's auc: 0.963632      h9's auc: 0.961277      h10's auc: 0.961023     h13's auc: 0.968868     h14's auc: 0.974623
[30]    train's auc: 0.97412    h4's auc: 0.962434      h5's auc: 0.965463      h9's auc: 0.963096      h10's auc: 0.962531     h13's auc: 0.969961     h14's auc: 0.975156
[40]    train's auc: 0.975427   h4's auc: 0.963675      h5's auc: 0.966746      h9's auc: 0.964093      h10's auc: 0.963756     h13's auc: 0.97079      h14's auc: 0.975685
[50]    train's auc: 0.976783   h4's auc: 0.964753      h5's auc: 0.967823      h9's auc: 0.965189      h10's auc: 0.964898     h13's auc: 0.971706     h14's auc: 0.976408

[115]   train's auc: 0.981139   h4's auc: 0.966736      h5's auc: 0.969638      h9's auc: 0.967221      h10's auc: 0.967047     h13's auc: 0.974154     h14's auc: 0.978856

[('lhood_ip_channel_hour', 7), ('lhood_ip_app_hour', 25), ('device', 70), ('count_ip_day_device_in_test_hh', 98), ('count_ip_day_hour', 114), ('count_ip_device_hour', 126), ('co
unt_ip_app_hour', 144), ('count_ip_day_app_in_test_hh', 157), ('count_ip_os_hour', 175), ('count_ip_day_in_test_hh', 218), ('hour', 316), ('os', 485), ('app', 551), ('channel', 
964)]


run10:
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        #'count_ip_app_channel_hour', 
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        'count_app_day_in_test_hh',
        #'delta_ip_device'
        #'lhood_ip_day_in_test_hh',
        #'lhood_app_channel_day_in_test_hh',
        #'lhood_ip_hour',
        #'lhood_ip_device_hour',        
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
        #'lhood_channel_hour',
        #'lhood_app_hour',
        #'lhood_app_channel_hour',
        #'lhood_os_hour'
    ]
[10]    train's auc: 0.968668   h4's auc: 0.957678      h5's auc: 0.960611      h9's auc: 0.95842       h10's auc: 0.95804      h13's auc: 0.965492     h14's auc: 0.971947
[20]    train's auc: 0.972548   h4's auc: 0.96153       h5's auc: 0.963954      h9's auc: 0.961619      h10's auc: 0.961142     h13's auc: 0.968787     h14's auc: 0.974717
[30]    train's auc: 0.974141   h4's auc: 0.962751      h5's auc: 0.965674      h9's auc: 0.9632        h10's auc: 0.962947     h13's auc: 0.969987     h14's auc: 0.974966
[40]    train's auc: 0.97551    h4's auc: 0.963978      h5's auc: 0.966964      h9's auc: 0.964108      h10's auc: 0.964154     h13's auc: 0.971282     h14's auc: 0.975743
[50]    train's auc: 0.97684    h4's auc: 0.964899      h5's auc: 0.968028      h9's auc: 0.965415      h10's auc: 0.965241     h13's auc: 0.972055     h14's auc: 0.976437
[60]    train's auc: 0.978188   h4's auc: 0.966052      h5's auc: 0.96883       h9's auc: 0.966201      h10's auc: 0.966062     h13's auc: 0.972977     h14's auc: 0.97731
[70]    train's auc: 0.979029   h4's auc: 0.966578      h5's auc: 0.969278      h9's auc: 0.966774      h10's auc: 0.966589     h13's auc: 0.973616     h14's auc: 0.97791
[80]    train's auc: 0.979693   h4's auc: 0.966871      h5's auc: 0.969585      h9's auc: 0.96711       h10's auc: 0.966865     h13's auc: 0.974148     h14's auc: 0.978396
[90]    train's auc: 0.98023    h4's auc: 0.967076      h5's auc: 0.969723      h9's auc: 0.967309      h10's auc: 0.966975     h13's auc: 0.974305     h14's auc: 0.97856
[100]   train's auc: 0.980688   h4's auc: 0.967064      h5's auc: 0.969718      h9's auc: 0.967215      h10's auc: 0.967141     h13's auc: 0.974449     h14's auc: 0.978711
[110]   train's auc: 0.981148   h4's auc: 0.967039      h5's auc: 0.969814      h9's auc: 0.967236      h10's auc: 0.967138     h13's auc: 0.974538     h14's auc: 0.978798
[120]   train's auc: 0.98148    h4's auc: 0.967133      h5's auc: 0.969872      h9's auc: 0.967277      h10's auc: 0.967235     h13's auc: 0.974602     h14's auc: 0.978839
[130]   train's auc: 0.981797   h4's auc: 0.967209      h5's auc: 0.970003      h9's auc: 0.967265      h10's auc: 0.967267     h13's auc: 0.97464      h14's auc: 0.978966
[140]   train's auc: 0.982167   h4's auc: 0.967246      h5's auc: 0.970019      h9's auc: 0.967291      h10's auc: 0.967313     h13's auc: 0.974685     h14's auc: 0.979
[150]   train's auc: 0.982499   h4's auc: 0.967252      h5's auc: 0.970074      h9's auc: 0.967322      h10's auc: 0.967367     h13's auc: 0.974686     h14's auc: 0.979049
[160]   train's auc: 0.982809   h4's auc: 0.967188      h5's auc: 0.970039      h9's auc: 0.967301      h10's auc: 0.967312     h13's auc: 0.974642     h14's auc: 0.979032
[170]   train's auc: 0.983063   h4's auc: 0.967181      h5's auc: 0.970029      h9's auc: 0.967283      h10's auc: 0.967287     h13's auc: 0.974605     h14's auc: 0.978988
[180]   train's auc: 0.983264   h4's auc: 0.967078      h5's auc: 0.969997      h9's auc: 0.967196      h10's auc: 0.967312     h13's auc: 0.974383     h14's auc: 0.978893
[190]   train's auc: 0.983446   h4's auc: 0.966993      h5's auc: 0.969889      h9's auc: 0.96707       h10's auc: 0.96721      h13's auc: 0.974127     h14's auc: 0.978705
Early stopping, best iteration is:
[147]   train's auc: 0.982412   h4's auc: 0.967279      h5's auc: 0.970081      h9's auc: 0.967342      h10's auc: 0.96735      h13's auc: 0.974733     h14's auc: 0.979053
[('lhood_ip_channel_hour', 19), ('count_app_day_in_test_hh', 70), ('device', 85), ('lhood_ip_app_hour', 110), ('count_ip_day_device_in_test_hh', 115), ('count_ip_device_hour', 169), ('count_ip_day_hour', 175), ('count_ip_day_app_in_test_hh', 200), ('count_ip_app_hour', 208), ('count_ip_day_in_test_hh', 265), ('count_ip_os_hour', 267), ('hour', 399), ('app', 574), ('os', 589), ('channel', 1165)]

run11:
    same as 10, setting lhood* as categorical.



[151]   train's auc: 0.982982   h4's auc: 0.967477      h5's auc: 0.969956      h9's auc: 0.968105      h10's auc: 0.968168     h13's auc: 0.975497     h14's auc: 0.979962
[('lhood_ip_channel_hour', 18), ('binip_lhood', 19), ('count_app_day_in_test_hh', 76), ('device', 79), ('lhood_ip_app_hour', 82), ('count_ip_day_device_in_test_hh', 106), ('count_ip_device_hour', 107), ('count_ip_day_hour', 138), ('count_ip_day_in_test_hh', 190), ('count_ip_day_app_in_test_hh', 201), ('count_ip_app_hour', 213), ('count_ip_os_hour', 271), ('hour', 372), ('binip', 416), ('app', 548), ('os', 574), ('channel', 1120)]
done  2018-04-06 17:43:19.266696


run12: x1 features with robustscaler (using day==9 to calculate target encoding)
Early stopping, best iteration is:
[197]   train's auc: 0.981368   h4's auc: 0.968385      h5's auc: 0.971294      h9's auc: 0.969402      h10's auc: 0.96942      h13's auc: 0.976414     h14's auc: 0.980387
[('count_app_day_in_test_hh', 23), ('count_ip_day_device_in_test_hh', 31), ('x1_app_device', 32), ('device', 35), ('count_ip_device_hour', 35), ('count_ip_day_hour', 41), ('count_ip_app_hour', 68), ('count_ip_os_hour', 81), ('x1_ip', 98), ('count_ip_day_app_in_test_hh', 99), ('x1_app_channel', 100), ('x1_app_os', 108), ('count_ip_day_in_test_hh', 135), ('hour', 268), ('app', 292), ('binip', 308), ('os', 334), ('channel', 670)]
done  2018-04-09 01:10:53.435381

run13: x1+x2 features with robustscaler (using day==9 to calculate target encoding)
[216]   train's auc: 0.981835   h4's auc: 0.968542      h5's auc: 0.971319      h9's auc: 0.969311      h10's auc: 0.968952     h13's auc: 0.976059     h14's auc: 0.980153
[('x2_ip_app_channel_var_day', 0), 
('x2_ip_app_day_hour_count_channel', 0), 
('x2_ip_day_channel_var_hour', 0), 
('x2_ip_day_hour_count_channel', 0), 
('x2_app_AvgViewPerDistinct_ip', 3), 
('x2_app_count_channel', 9), 
('x2_channel_count_app', 11), 
('x2_ip_app_channel_mean_hour', 14), 
('count_app_day_in_test_hh', 15), 
('x2_ip_app_count_channel', 21), 
('x2_ip_app_os_count_channel', 28), 
('x1_app_device', 30), 
('x2_ip_app_os_var_hour', 32), 
('device', 36), 
('count_ip_device_hour', 37), 
('count_ip_day_device_in_test_hh', 39), 
('count_ip_day_hour', 51), 
('count_ip_day_app_in_test_hh', 77), 
('count_ip_app_hour', 80), 
('count_ip_os_hour', 85), 
('x1_ip', 86), 
('x1_app_channel', 106), 
('x1_app_os', 109), 
('count_ip_day_in_test_hh', 137), 
('app', 290), 
('hour', 297), 
('binip', 342), 
('os', 350), 
('channel', 739)]

params = {
    'num_leaves': 15,
    'max_depth': 5,    
    'max_bin': 255,
}


[102]   train's auc: 0.982155   h4's auc: 0.968016      h5's auc: 0.971814      h9's auc: 0.969552      h10's auc: 0.969672     h13's auc: 0.976022     h14's auc: 0.980312
[('x2_ip_app_os_var_hour', 3), ('x2_app_count_channel', 7), ('count_ip_app_channel_hour', 10), ('x2_ip_app_count_channel', 10), ('x2_ip_app_channel_mean_hour', 11), ('count_ip_app_hour', 13), ('
x2_ip_app_os_count_channel', 17), ('device', 22), ('x1_app_device', 25), ('x1_app_os_hour', 26), ('x1_app_channel_hour', 27), ('x1_ip', 27), ('count_ip_os_hour', 37), ('count_ip_device_hour', 47
), ('count_ip_day_app_hour', 52), ('hour', 53), ('x1_app_os', 59), ('x1_app_channel', 70), ('count_ip_day_hour', 109), ('app', 115), ('binip', 162), ('os', 176), ('channel', 350)]
done  2018-04-10 02:39:24.828795


run13: added x3
[86]    train's auc: 0.981497   h4's auc: 0.969227      h5's auc: 0.972491      h9's auc: 0.970777      h10's auc: 0.970672     h13's auc: 0.9774       h14's auc: 0.981656
[('x3_binip', 0), ('x2_ip_app_channel_mean_hour', 1), ('x2_app_count_channel', 2), ('x2_ip_app_count_channel', 2), ('x2_ip_app_os_var_hour', 2), ('x2_ip_app_os_count_chann
el', 6), ('count_ip_app_channel_hour', 7), ('count_ip_app_hour', 7), ('x3_binip_app_channel_hour', 11), ('x1_app_os_hour', 14), ('x3_binip_hour', 14), ('x3_binip_app_os', 
15), ('device', 18), ('x1_app_channel_hour', 19), ('x3_binip_app_os_hour', 20), ('x1_app_channel', 21), ('x1_app_device', 22), ('count_ip_os_hour', 23), ('x1_ip', 23), ('h
our', 33), ('count_ip_device_hour', 37), ('x1_app_os', 44), ('count_ip_day_app_hour', 50), ('x3_binip_app_channel', 61), ('app', 97), ('count_ip_day_hour', 97), ('binip', 
126), ('os', 147), ('channel', 285)]

run14: added x4 with all days except 8 ((df.day.isin([6, 7, 9])) & df.hour.isin([4,5,9,10,13,14]))

[85]    train's auc: 0.981486   h4's auc: 0.969668      h5's auc: 0.973027      h9's auc: 0.971214      h10's auc: 0.971095     h13's auc: 0.977906     h14's auc: 0.982335
[('x2_app_count_channel', 0), ('x2_ip_app_count_channel', 0), ('x2_ip_app_os_var_hour', 2), ('x2_ip_app_channel_mean_hour', 4), ('x2_ip_app_os_count_channel', 6), ('count_ip_app_channel_hour', 10), ('count_ip_app_hour', 10), ('x4_binip_hour', 10), ('x1_app_os_hour', 16), ('x4_binip_app_channel_hour', 17), ('count_ip_os_hour', 18), ('x1_app_channel_hour', 18), ('x4_binip_app_os', 18), ('device', 19), ('x1_app_channel', 21), ('x1_ip', 22), ('x1_app_device', 23), ('count_ip_device_hour', 27), ('x4_binip_app_os_hour', 31), ('hour', 35), ('x1_app_os', 46), ('count_ip_day_app_hour', 54), ('x4_binip_app_channel', 58), ('app', 84), ('count_ip_day_hour', 102), ('binip', 119), ('os', 132), ('channel', 288)]
"""


import os
import gc
import pickle
import logging
import datetime
import operator
from datetime import datetime 

import pandas as pd
import numpy as np
import lightgbm as lgb

import data2
import train2 

from dataset import *
from util import info

SEED = 0

from sklearn.preprocessing import RobustScaler 

if __name__ == '__main__':
    trainval_df = data2.load('train')
    
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'binip']

    # faster feedback
    train_cond = (trainval_df.day == 8) & (trainval_df.hour.isin([4,5,9,10,13,14]))
    train_df = trainval_df[train_cond] 
    #train_df = trainval_df.iloc[:-VALID_ROWS]
        
    #info('shuffling train')
    #train_df = train_df.iloc[np.random.permutation(len(train_df))]
        
    # used to save memory only, as when building lgbm dataset we specify
    # columns to be used explicitly
    excluded = [
        'click_time',
        'ip',
        'day'
    ]
    
    for column in excluded:
        del trainval_df[column]
        
    #apply scaler to x1
    
    predictors = list(sorted([c for c in trainval_df.columns if c not in ['is_attributed']]))
    
    gc.collect()
             
    val_df = trainval_df.iloc[-VALID_ROWS:]
    
    val_dfs = {
        'h4':  val_df[val_df.hour ==  4],
        'h5':  val_df[val_df.hour ==  5],
        'h9':  val_df[val_df.hour ==  9],
        'h10': val_df[val_df.hour == 10],
        'h13': val_df[val_df.hour == 13],
        'h14': val_df[val_df.hour == 14]
    }
    
    params = {
        'num_leaves': 15,
        'max_depth': 5,    
        'max_bin': 255
    }
    
    info('predictors: %s' % ','.join(predictors))
    
    iterations = 500
    train2.run(train_df, val_dfs, 
        predictors, target, categorical, 
        iterations, SEED, param_overrides=params)
    
    
