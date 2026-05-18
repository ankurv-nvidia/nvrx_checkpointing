1. Start initial run
2. Take a distcp checkpoint [dcp0]
3. Take a native checkpoint [ntv0]

4. Iter-1 - Take a distcp checkpoint.[dcp1]
5. Iter-1 - Take a native checkpoint [ntv1]
6. Stop
7. Load from iter-1 native checkpoint [ntv0]
8. Iter-2 - Take a native checkpoint [ntv2]
9. Iter-2 - Take a distcp checkpoint [dcp2]

ntv0 == initial-data
ntv1 == ntv2
dcp1 == dcp2 
dcp0 tensors === initial-data
