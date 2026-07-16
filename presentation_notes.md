# Script

1. **Title** — Today we're presenting our reinforcement learning and strategy-based model for playing Balatro.

2. **What is Balatro?** — Balatro is a roguelike deckbuilding card game where you score poker hands to beat increasingly difficult rounds. You can buy unique Jokers or other kinds of modifiers every round that modify how your score is calculated.

3. **Core Problem** — The core challenge is that a Joker's value depends entirely on what else you own, and success requires planning across 8 antes, each ante has 3 rounds, every 3rd round has a boss that has different kind of modifiers on the round making it harder to win. Because amount of modifiers are too high, RL agents tend to fail to reach high antes.

4. **Environment Classification** — The environment is partially observable because agent can't see the future. We used "seeds", which makes the randomness of the game deterministic. The game is based partially on poker but it is a singleplayer game with no opponent, you just go through 24 rounds to try to win the game, so it is single-agent. It is also turn-based. But at the end, it still forms a huge decision space once you factor in 150 Jokers and the shop economy.

5. **Two-Tier Architecture** — We noticed that most experienced players play the game in 2 ways. First, they have a defined strategy of going for flush build, a pair build or a mult build, then they try to buy modifiers according to that. Having noticed that, we decided to follow a 2 tier architecture where there is a strategic layer formed by an RL model that picks a high-level build and buys jokers according to that, and the other tier is a heuristics based operational layer that handles actual card play and discarding, mirroring how expert players think.

6. **RL Model Architecture** — Our model is a PPO agent with a 25-unit input layer, two 128-unit hidden layers, and a 10 and 3 multi-discrete output that handles shop actions and strategy selection together.

7. **Reward Shaping** — We decided to keep the reward shaping very simple. We reward the agent more the further it progresses through antes, and penalize it if it loses at the first ante or for selling Jokers.

8. **Decision Point #1: Platform** — We initially planned to build upon balatro-gym, a library that supposedly would allow us to build and play around with RL in a Balatro emulator, but we got a lot of problems trying it. There were a lot of features of the game missing and the code would frequently crash. We abandoned it. We decided to use Balatrobot v1.4.1 for RPC control over the real game engine and train the model on the actual game.

9. **The Mod** — As we have mentioned, the game has a lot of modifiers and overall complexity in it, which makes it hard to start training a model and observe how well it performs, there is just too much RNG and modifiers going on. To make training feasible, we built a mod that simplifies the game down to just 14 Jokers, removes other consumables, and disables boss blind effects. We also defined seeds to make the state determinent.

10. **Decision Point #2: Strategy → Shop** — First, we had the model pick just one strategy label per round, later we also added the shop decisions as well to it. The first version we had was playing the cards on a heuristic based approach, but it never discarded the cards. After a bit of a research, we found out that Monte Carlo method would be a good way to implement discarding so we implemented that.

11. **Monte Carlo Discards** — A deterministic calculator can't judge whether discarding worths more than playing the cards the player has. With Monte Carlo method, we simulate what cards the player can get from the deck and what score he could achieve. We defined the limit of number of simulations as 50 but that caused our training to become 6 times longer. So we optimized it with a caching mechanism and a short-circuit pruning where it stops calculating the other combinations if it is pre-calculated that it is not worth it. With that, it the training take only around 15% longer, which wasn't big of an issue.

12. **Infra Speedups** — We were training on the actual game, so it would take almost 36 hours to train. By speeding up episode resets and running four parallel instances, we cut full training time from about 36 hours down to about 8 hours.

13. **Decision Point #3: Emulator** — We were already using a headless version of the game so no graphic interface was present, but after seeing the projects of other groups and getting feedback from Mr Ruscheinski, we wanted to use an emulator. We found a repo called Jackdaw that had a Balatro emulator. So we integrated that emulator, which reproduces the game on Python without any graphics or network overhead.

14. **Emulator Results** — The emulator made training about 12 times faster, about 30-40 minutes, and we integrated it in such a way that the training script only uses an adapter between the actual game and the emulator, so the whole training script is easily switched between the game and the emulator. This training time speed up allowed us to better observe how our model works, and with that we also tried and see how it works in full, non-simplified game environment.

15. **Replay System** — But sometimes the model would reach higher antes, sometimes it would fail to reach even ante 4. We had the state of the deck being displayed on the terminal but that was realtime, we needed something different that would allow us to see what exactly has happened during the game. We first build web-based dashboard to play the emulator, then we also added a replay system to step through any training episode and see exactly what the agent bought, played, and held at each point.

16. **Latest Infrastructure** — These are the features that we managed to put out in during the project. (Swiftly read the stuff on the slide)

17. **Results** — To see how well our models perform, we tried a combination of environment and seed count to train on. Compared to the models that were trained on a single seed and 100 seeds, the models trained on 1 million seeds generalized best with the highest average ante of around 4.2, but that is on a simplified environment. Compared to the simple environment, the models trained on the full environment reached a higher maximum ante and even won by hitting ante 11, but they also had a lower average ante count. We think the reasons for this are lack of multiplicative Jokers in simplified environment limiting it to the ante 6, and inclusion of the boss blind modifiers in the full environment, which makes it harder to beat earlier antes.

18. **Summary** — We concluded that the performance plateau was a limitation of the environment rather than the agent. We have compared our agent to the projects. The model that was in Jackdaw managed to only reach to 3rd ante after 400k steps, and it also crashed after that. Our model manages to reach some kind of plateu after 10k steps and manages to occasionally beat the game as well, even without having full access the shop such as tarrots and planets and stuff. We think that further improvements can be achieved like this: (read the 3 improvements parts)

19. **Thanks** — This is our agent trained on emulator playing the actual game. Thank you for your attention.



Keypoints:


- We have only 14 defined jokers, because simplified version only has those. But full version has more jokers, we don't know how our model sees this. Teacher should not know about this.
- (It should be implied to the reader that every decision in this project was preceded by a concrete finding and research - a benchmark, a bug, or a plateau - not a preference.)