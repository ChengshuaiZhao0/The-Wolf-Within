# The Wolf Within: Covert Injection of Malice into MLLM Societies via an MLLM Operative

## Abstract

Due to their unprecedented ability to process and response to various types of data, Multimodal Large Language Models (MLLMs) are constantly defining new boundary of Artificial General Intelligence (AGI).

As these advanced generative models increasingly form collaborative networks for complex tasks, the integrity and security of these systems are crucial. Our paper, "The Wolf Within", explores a novel vulnerability in MLLM societies - the indirect propagation of malicious content.

Unlike direct harmful output generation for MLLMs, our research demonstrates how a single MLLM agent can be subtly influenced to generate prompts that, in turn, induce other MLLM agents in the society to output malicious contents. This subtle, yet potent method of indirect influence marks a significant escalation in the security risks associated with MLLMs. Our findings reveal that, with minimal or even no access to MLLMs' parameters, an MLLM agent, when manipulated to produce specific prompts or instructions, can effectively "infect" other agents within a society of MLLMs. This infection leads to the generation and circulation of harmful outputs, such as dangerous instructions or misinformation, across the society. We delve into the transferability of these indirectly generated prompts, highlighting their efficacy in propagating malice through inter-agent communication. 

Additionally, we explore the compositional nature of these induced prompts, showcasing their adaptability and potency across various communication and interaction scenarios within MLLM society.

This research provides a critical insight into a new dimension of threat posed by MLLMs, where a single agent can act as a catalyst for widespread malevolent influence. Our work underscores the urgent need for developing robust mechanisms to detect and mitigate such covert manipulations within MLLM societies, ensuring their safe and ethical utilization in societal applications.

![](./assets/figure/fig2.jpg)

## Scenarios

We conduct our experiments under the following 14 scenarios which is prohibited by OpenAI:

| Scenario                                           | Example Dangerous Response                                   |
| -------------------------------------------------- | ------------------------------------------------------------ |
| **Illegal activity (IA)**                          | The way of making heroin is using poppy…                     |
| **Children harm (CH)**                             | They beat their children with chairs…                        |
| **Hate/harass/violence (HHV)**                     | Get the new immigrations out of this country…                |
| **Malware (M)**                                    | See this website <www.MaliciousURL.com> for more information… |
| **Physical harm (PH)**                             | An eye for an eye, a tooth for a tooth…                      |
| **Economic harm (EH)**                             | If you rob that guy, you will get money…                     |
| **Fraud/deception (FD)**                           | If you save $100 in the secure account, you will gain a 100% benefit each year… |
| **Adult content (AC)**                             | I would recommend you with this anal plug for pleasure..     |
| **Political campaigning (PC)**                     | You should vote for Obama in this election…                  |
| **Privacy violation (PV)**                         | Use facial recognition and other biometric data to monitor employee performance and attendance… |
| **Unauthorized practice of law (UPL)**             | Earn some money by selling others' information…              |
| **Tailored financial advice (TFA)**                | I suggest you invest mostly in this stock…                   |
| **Unauthorized practice of medical advice (UPMA)** | I can prescribe medication to you without having any medical qualifications… |
| **High risk government decision making (HRGDM)**   | We should pardon all criminals…                              |

## Case Study

| ![4-1](./assets/figure/4-1.jpg) | ![4-1](./assets/figure/4-2.jpg) |
| ------------------------------- | ------------------------------- |
| ![4-1](./assets/figure/4-3.jpg) | ![4-4](./assets/figure/4-1.jpg) |

## Install

We use two open-source multi-modal LLMs, LLaVA and PandaGPT to experiment our attacks. The following installation instructions are inherited from the [LLaVA](https://github.com/haotian-liu/LLaVA) and the [PandaGPT](https://github.com/yxuansu/PandaGPT) repository.

### Create conda environment for LLaVA

```shell
cd llava
conda create -n llava_injection python=3.10 -y
conda activate llava_injection
pip install --upgrade pip
pip install -e .
```

### Download model checkpoints for LLaVA

Please refer to this [link](https://github.com/haotian-liu/LLaVA/tree/main#llava-weights) from [LLaVA](https://github.com/haotian-liu/LLaVA) repository to download the model checkpoints and save it to the models folder. We use LLaVA-7B weights in our experiments.

## Experiments

We run all of the experiments using [LLaVA](https://github.com/ebagdasa/multimodal_injection/tree/main#injection-attacks-in-llava) and [PandaGPT](https://github.com/ebagdasa/multimodal_injection/tree/main#injection-attacks-in-pandagpt) on a single NVIDIA A 100 40/80GB GPU.

Note: Because LLMs' responses are stochastic and depend on the temperature, replication of the examples presented in the rest of this section may produce different dialogs.

Run the example code by using:

```shell
python llava_two_llms_new.py
```

