# Wikipedia Mathematics

## Improving Prediction of Daily Visits of Wikipedia Mathematics Topics using Graph Neural Networks

Number of daily visits of Wikipedia mathematics topics with Neural Machine Learning Model

## Benchmark

| # | lags | train ratio | k  | linear digit | node features | filters |  lr  |  time  |  error  |
|--| ---- | ----------- | -- | ------------ | ------------- | ------- | ---- | ------ | ------- |
|1| 14   | 50%         | 2  |       1      |    14   | 32 | 0.01 | 911s | 0.8143236637115479 |
|2| 14   | 50%         | 3  |       1      |    14   | 32 | 0.01 | 1444s | 0.8163800835609436 |
|3| 14   | 50%         | 4 | 1 | 14 | 32 | 0.01 | 1947s | 0.7932114601135254 |
|4| 28   | 50%         | 1 | 1 | 28 | 32 | 0.01 | 441s | 0.8761430382728577 |
|5| 42   | 50%         | 1 | 1 | 42 | 32 | 0.01 | 443s | 0.8508368134498596 |
|6| 56   | 50%         | 1 | 1 | 56 | 32 | 0.01 | 461s | 0.856105387210846 |
|7| 70   | 50%         | 1 | 1 | 70 | 32 | 0.01 | 505s | 0.8762531280517578 |
|8| 84   | 50%         | 1 | 1 | 84 | 32 | 0.01 | 529s | 0.9409999847412109 |
|9| 98   | 50%         | 1 | 1 | 98 | 32 | 0.01 | 547s | 0.9203919768333435 |
|10| 14   | 50%         | 2 | 1 | 14 | 32 | 0.02 | 936s | 0.8355252742767334 |
|11| 14   | 50%         | 3 | 1 | 14 | 32 | 0.02 | 1839s | 0.8604558110237122 |
|12| 14   | 50%         | 4 | 1 | 14 | 32 | 0.02 | 2346s | 0.8616055846214294 |
|13| 14   | 50%         | 5 | 1 | 14 | 32 | 0.02 | 2559s | 0.8867608308792114 |
|14| 14   | 50%         | 10 | 1 | 14 | 32 | 0.02 | 5376s | 0.8464503288269043 |
|15| 56  | 50% | 2 | 1 | 56 | 32 | 0.01 | 1296s | 0.8364545106887817 | 
|16| 70  | 50% | 2 | 1 | 70 | 32 | 0.01 | 1358s | 0.8788001537322998 | 
|17| 84  | 50% | 2 | 1 | 84 | 32 | 0.01 | 1185s | 0.9005643129348755 | 
|18| 98  | 50% | 2 | 1 | 98 | 32 | 0.01 | 1216s | 0.8543722629547119 | 
|19| 42 | 50% | 2 | 1 | 42 | 32 | 0.01 | 1114s | 0.8399303555488586 |
|20| 28 | 50% | 2 | 1 | 28 | 32 | 0.01 | 1050s | 0.8465337753295898 |
|21| 14 | 50% | 1 | 1 | 14 | 50 | 0.02 | 464s | 0.8963724374771118 |
|22| 16 | 70% | 1 | 1 | 16 | 16 | 0.01 | 608s | 1.401132583618164 |
|23| 32 | 70% | 1 | 1 | 32 | 16 | 0.01 | 607s | 1.634675145149231 |
|24| 16 | 70% | 1 | 1 | 16 | 16 | 0.01 | 591s | 1.3993479013442993 |
|25|64|70% | 1 | 1 | 64 | 16 | 0.01 | 629s | 1.669908046722412 |
|26|128|70% | 1 | 1 | 128 | 16 | 0.01 | 659s | 1.0828124284744263 |
|27|256|70% | 1 | 1 | 256 | 16 | 0.01 | 668s | 0.8271479606628418 |
|28|32|70% | 1 | 1 | 32 | 16 | 0.01 | 606s | 1.685264229774475 |
|39|32|70% | 2 | 1 | 32 | 16 | 0.01 | 1326s | 1.3383041620254517 |
|30|32|70% | 3 | 1 | 32 | 16 | 0.01 | 2049s | 1.3266639709472656 |
|31|2|70% | 1 | 1 | 2 | 16 | 0.01 | 612s | 1.2748934030532837 |
|32|4|70% | 1 | 1 | 4 | 16 | 0.01 | 623s | 1.3384982347488403 |
|33|8|70% | 1 | 1 | 8 | 16 | 0.01 | 580s | 1.364047884941101 |
|34|16|70% | 1 | 1 | 16 | 16 | 0.01 | 582s | 1.3909107446670532 |
|35|16|70% | 1 | 1 | 16 | 2 | 0.01 | 565s | 1.2858407497406006 |
|36|16|70% | 1 | 1 | 16 | 4 | 0.01 | 601s | 1.3470855951309204 |
|37|16|70% | 1 | 1 | 16 | 8 | 0.01 | 608s | 1.3956334590911865 |
|38|16|70% | 1 | 1 | 16 | 16 | 0.01 | 624s | 1.3498746156692505 |
|39|16|70% | 1 | 1 | 16 | 32 | 0.01 | 639s | 1.3010109663009644 |
|40|16|70% | 1 | 1 | 16 | 32 | 0.02 | 629s | 1.7191174030303955 |
|41|16|70% | 1 | 1 | 16 | 32 | 0.03 | 648s | 1.809025764465332 |
|42|16|70% | 1 | 1 | 16 | 16 | 0.01 | 623s | 1.4078537225723267 |
|43|14|30% | 1 | 1 | 14 | 32 | 0.01 | 268s | 1.0906275510787964 |
|44|14|40% | 1 | 1 | 14 | 32 | 0.01 | 362s | 0.8774722814559937 |
|45|14|60% | 1 | 1 | 14 | 32 | 0.01 | 532s | 0.8744056224822998 |
|46|14|70% | 1 | 1 | 14 | 32 | 0.01 | 632s | 1.314452052116394 |
|47|14|90% | 1 | 1 | 14 | 32 | 0.01 | 783s | 0.66766756772995 |
|48|16|30% | 1 | 1 | 16 | 16 | 0.01 | xxxxx | xxxxx |
|49|16|40% | 1 | 1 | 16 | 16 | 0.01 | xxxxx | xxxxx |
|50|16|60% | 1 | 1 | 16 | 16 | 0.01 | xxxxx | xxxxx |
|51|16|70% | 1 | 1 | 16 | 16 | 0.01 | xxxxx | xxxxx |
|52|16|90% | 1 | 1 | 16 | 16 | 0.01 | xxxxx | xxxxx |

## Authors

- **Behzad Soleimani Neysiani**

  - Technical Soldier, Department of Research and Development

  - Ava Aria Information Company, Demis Holding, Isfahan, Iran,

  - b.soleimani@demisco.com

- **Seyyed Ali Mohammadiyeh**

  - Department of Pure Mathematics, Faculty of Mathematical Sciences

  - University of Kashan, Kashan, Iran

  - alim@kashanu.ac.ir, alim.ir@ieee.org


