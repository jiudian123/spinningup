import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.tf1.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) 
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.存入经验池
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)  #元组切片 [path_start_idx，ptr)，长度为1,数值从0递增
        rews = np.append(self.rew_buf[path_slice], last_val) #在self.rew_buf[path_slice]加入last_val
        vals = np.append(self.val_buf[path_slice], last_val) #同上
        
        # the next two lines implement GAE-Lambda advantage calculation 实际V值
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1] #deltas（J）=Rt+1 + gamma*Vt+1 - Vt,  rews[:-1]从位置0到位置-1之前的数
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function 目标V值
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.取出经验池所有数据
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick 归一化
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf] 



def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.策略给出的动作
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.策略给出的x状态下的a动作概率
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.动作被采样的概率
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)状态x下的V值
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO. AC框架参数

        seed (int): Seed for random number generators.#随机种子数0
        
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.每轮迭代次数4000

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.轮次50

        gamma (float): Discount factor. (Always between 0 and 1.)折扣因子0.99

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 剪切比率0.2

        pi_lr (float): Learning rate for policy optimizer.策略学习率3e-4

        vf_lr (float): Learning rate for value function optimizer.评价网络学习率1e-3

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)策略最大梯度下降步数80

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.评价最大梯度下降步数80

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.) TD（lambda）中的lambda =0.97

        max_ep_len (int): Maximum length of trajectory / episode / rollout. 每次最长步长 1000

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)提前停车用，目标kl 0.01

        logger_kwargs (dict): Keyword args for EpochLogger.日志参数

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.模型保存频率每10轮

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn() #创建环境
    obs_dim = env.observation_space.shape #读取环境维度
    act_dim = env.action_space.shape #动作维度
    
    # Share information about action space with policy architecture 策略框架下共享动作有关信息 
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs) #输入环境和动作，输出策略相关信息

    # Need all placeholders in *this* order later (to zip with data from buffer)之后需要的数据：环境，动作，优势函数、奖励和上一步的策略
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob每步得到的信息：策略、V值和动作概率
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam) #经验池

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v']) #记录变量
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts) #日志输出策略和评价的变量数

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s) 比率=pi（new）/pi（old）
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph) #tfwhere条件adv_ph,>0，表示优势增加，选(1+clip_ratio)*adv_ph，
    #<0，表示优势减少，选(1-clip_ratio)*adv_ph。
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv)) #策略loss=比率adv与min_adv的最小者，限制策略偏移
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute  KL散度的样本估计，易于计算
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute  熵的样本估计，易于计算
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio)) # 逻辑或运算，判定需要剪切，clipped = true/false
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))  #clipped转换为浮点数

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())} # zip（[x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]，
        #[self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf] ）
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs) # 输入上述数据，计算俩loss和熵

        # Training
        for i in range(train_pi_iters):#策略迭代
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs) # 计算kl
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break  #提前停止策略训练
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)# 训练评价网络

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs) #重新计算loss和kl，cf
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))  # 输出旧loss，kl，cf 和 delta loss

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0 #重置

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs): # 每一轮
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})  #根据环境给出动作

            o2, r, d, _ = env.step(a[0]) #环境交互取得奖励
            ep_ret += r #奖励累加
            ep_len += 1 # 步长加一 

            # save and log
            buf.store(o, a, r, v_t, logp_t) #存储（环境、动作、奖励、V值、概率）经验池
            logger.store(VVals=v_t)

            # Update obs (critical!)更新环境
            o = o2

            terminal = d or (ep_len == max_ep_len) #结束了或者到达最大步数了
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len) #轨迹被epoch在ep_len步切断
                # if trajectory didn't reach terminal state, bootstrap value target  如果轨迹没有达到终端状态，引导值目标
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
