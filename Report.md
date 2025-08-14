## Problem Statement

#### We studied a practical problem: how to train useful, private AI models across lots of different edge devices (phones, sensors, small servers) that all have different data, speeds and network links. In real life, these devices don’t share the same kind of data, some are slow, and bandwidth is limited — so training one single global model and inviting every device to train every time just doesn’t work well.

## Solution

#### To address this, we implemented the approach from the paper: two things working together. First, we pick which devices participate each round based on an estimate of how long they’ll take (compute + upload). That keeps rounds fast while still rotating slower devices in so they aren’t ignored. Second, we group devices whose updates look similar — using gradient similarity — and train a separate model for each group. The result is that each group gets a model that fits its data better.

## Idea of Implementation

#### We built a simulated environment in Python using SimPy and the FEMNIST dataset. Each client is a simulated process with its own data shard, CPU speed and channel conditions. Clients train a small CNN locally and send only model deltas to the edge server. The edge server schedules clients, aggregates updates (FedAvg), and runs the clustering logic to split or maintain clusters. We log accuracy, round time, and participation to evaluate performance.

## Limitation

#### We’re realistic about limitations. This is a simulation, not a hardware deployment, so real-world wireless noise, device failures, or energy constraints need extra validation. The clustering thresholds need careful tuning (too aggressive = over-splitting; too conservative = under-splitting). There’s also room to reduce communication cost further

## Q & A

### 1.How does an Edge server differ from Cloud server and how does it impact in this study?

#### So, an edge server is something we call the master edge node, which is placed closer and connects N number of edge devices. It's basically where the edge devices are controlled or where the higher-end, complex ML operations are handled. Now, coming to cloud servers, sending all the data to the cloud for processing isn’t really practical, because it increases the overall latency of the system. So, in this study, instead of relying on cloud servers, the edge servers take on the responsibility of clustering or grouping the edge devices based on their similarity. This not only reduces latency but also ensures data privacy is maintained.

### 2.You say that the server will continuously train the models to send to the cluster or group of edge devices. How and when will that training stop?

#### As we mentioned, the process of training the models on both the edge devices and the edge server appears to be continuous. However, there comes a stationary point where the data or the number of clusters/groups becomes zero or gets exhausted. At this point in training, it stops and waits for the new set of data from the edge device.

### 3.How do you say that the devices will perform the processing of ML?

#### The Edge devices mentioned in the paper refers to devices at the field this system works. The devices would be an smart phone or an RasberyPi depending on the scenario or use case this works.So Depending upon the scenario and the use case the ML and performance of Ml depands on the operation it is gona perform.

### 4. How do define the parameters for Cosine similarity?

#### 

