### Continuous Concurrent Credit Assignment (CCCA)

The code follows the logic of the spinning up in deep RL library (https://spinningup.openai.com) and can be installed as an ''add-on'' algorithm to the existing ones. (It can also be used stand-alone)

It is supposed to run on continuous environments under the gym suite : https://www.gymlibrary.ml/

**Example:**

    #!/bin/csh
    set env_name=ant
    set envir=Ant-v2
    set BETA_REF=10 # reward amplification
    set PREC=1
    set gamma=0.99
    set bandwidth=0.3
    set start_steps=10000
    set steps_per_epoch=4000
    set update_after=10000
    set update_every=50
    set epochs=500
    set replay_size=1000000 #int(1e6)
    set batch_size=100
    set max_ep_len=1000
    set hid=256
    python cca --hid "[$hid,$hid]" --env $envir --exp_name ant_cca_test --beta $BETA_REF --prec $PREC --gamma $gamma --bandwidth $bandwidth --epochs $epochs --steps_per_epoch $steps_per_epoch --start_steps $start_steps  --update_after $update_after --update_every $update_every --replay_size $replay_size --batch_size $batch_size --max_ep_len $max_ep_len&
