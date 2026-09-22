Table T1. The 20 classes of mixalldata_clean.csv with the measurements that identify them (all computed from the data).

| class_id | name | group | messages | share_of_messages_% | vehicles | median_seconds_between_messages | median_pseudonyms_per_vehicle | mean_speed_m_per_s | share_speed_zero_% | share_identical_to_another_message_% | share_copy_of_a_genuine_message_% | definition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Genuine | genuine | 1900539 | 59.4900 | 17264 | 1.0000 | 1.0000 | 9.7100 | 0.2000 | 6.7000 | 0.0000 | Normal behaviour: the vehicle broadcasts its real position, speed, acceleration and heading. |
| 1 | Constant position | fault | 43653 | 1.3700 | 390 | 1.0000 | 1.0000 | 9.7300 | 0.1000 | 9.3000 | 0.0000 | The transmitted position stays constant (frozen) while the vehicle keeps moving. |
| 2 | Constant position offset | fault | 43567 | 1.3600 | 390 | 1.0000 | 1.0000 | 9.8200 | 0.1000 | 3.7000 | 0.0000 | The transmitted position is the real position plus a fixed offset. |
| 3 | Random position | fault | 43857 | 1.3700 | 390 | 1.0000 | 1.0000 | 9.5500 | 0.2000 | 3.9000 | 0.0000 | The transmitted position is drawn uniformly at random from the simulation playground in every message. |
| 4 | Random position offset | fault | 42575 | 1.3300 | 389 | 1.0000 | 1.0000 | 9.6500 | 0.1000 | 6.9000 | 0.0000 | The transmitted position is the real position plus a random offset that changes in every message. |
| 5 | Constant speed | fault | 41925 | 1.3100 | 390 | 1.0000 | 1.0000 | 31.0800 | 0.0000 | 5.2000 | 0.0000 | The transmitted speed stays constant regardless of the vehicle's real motion. |
| 6 | Constant speed offset | fault | 44359 | 1.3900 | 389 | 1.0000 | 1.0000 | 13.6300 | 0.0000 | 6.8000 | 0.0000 | The transmitted speed is the real speed plus a fixed offset. |
| 7 | Random speed | fault | 42258 | 1.3200 | 389 | 1.0000 | 1.0000 | 30.6200 | 0.0000 | 5.9000 | 0.0000 | The transmitted speed is drawn uniformly at random in every message. |
| 8 | Random speed offset | fault | 42583 | 1.3300 | 389 | 1.0000 | 1.0000 | 13.7200 | 0.0000 | 6.9000 | 0.0000 | The transmitted speed is the real speed plus a random offset that changes in every message. |
| 9 | Eventual stop | attack | 42790 | 1.3400 | 389 | 1.0000 | 1.0000 | 1.8400 | 80.9000 | 82.4000 | 0.0000 | The vehicle simulates a sudden stop by freezing its transmitted position and setting its speed to zero. |
| 10 | Disruptive | attack | 43264 | 1.3500 | 390 | 1.0000 | 1.0000 | 13.0700 | 2.0000 | 99.1000 | 63.1000 | The vehicle re-broadcasts data previously received from random neighbouring vehicles as if it were its own. |
| 11 | Data replay | attack | 44337 | 1.3900 | 389 | 1.0000 | 1.0000 | 10.4600 | 1.1000 | 99.3000 | 80.1000 | The vehicle re-broadcasts data previously received from one specific target neighbour as if it were its own. |
| 12 | Delayed messages | fault | 43118 | 1.3500 | 390 | 1.0000 | 1.0000 | 4.8000 | 51.3000 | 53.3000 | 0.0000 | Messages contain correct data but are sent with a delay, so the reported state is stale. |
| 13 | DoS | attack | 131305 | 4.1100 | 390 | 0.2500 | 1.0000 | 9.7300 | 0.1000 | 99.9000 | 0.0000 | Denial of service: the vehicle sends messages at a frequency higher than the limit set by the standard. |
| 14 | DoS random | attack | 126724 | 3.9700 | 389 | 0.2500 | 1.0000 | 30.6400 | 0.0000 | 1.7000 | 0.0000 | Denial of service in which all message fields are additionally set to random values. |
| 15 | DoS disruptive | attack | 129270 | 4.0500 | 389 | 0.2500 | 1.0000 | 12.5400 | 1.6000 | 99.9000 | 66.7000 | Denial of service in which the high-frequency messages replay data received from random neighbours. |
| 16 | Grid Sybil | attack | 175391 | 5.4900 | 389 | 0.1670 | 6.0000 | 2.9900 | 0.9000 | 4.6000 | 0.0000 | Traffic-congestion Sybil attack: the attacker creates a grid of ghost vehicles, each with its own pseudonym. |
| 17 | Data replay Sybil | attack | 44310 | 1.3900 | 389 | 1.0000 | 57.0000 | 10.6600 | 0.8000 | 99.1000 | 78.9000 | Data replay in which the attacker switches to a new pseudonym for every new target vehicle. |
| 18 | DoS random Sybil | attack | 86883 | 2.7200 | 390 | 0.5000 | 100.0000 | 30.6400 | 0.0000 | 13.0000 | 0.0000 | DoS random in which the attacker changes its pseudonym on every message. |
| 19 | DoS disruptive Sybil | attack | 82100 | 2.5700 | 389 | 0.5000 | 100.0000 | 11.9200 | 2.2000 | 99.9000 | 68.2000 | DoS disruptive in which the attacker changes its pseudonym on every message. |
