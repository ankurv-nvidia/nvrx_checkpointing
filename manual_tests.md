# oci-hsg

git -C ~/data/megatron-lm/ pull; rm -rf ~/data/megatron-lm/runs/checkpoints/; bash ~/data/megatron-lm/megatron/native/scripts/enroot.sh run |& tee /tmp/out.log |& grep -E 'native_ckpt.*perf' > ~/data/megatron-lm/megatron/native/perf-rslt-ocihsg.log &

# cw-dfw

git -C ~/data/megatron-lm/ pull; rm -rf ~/data/megatron-lm/runs/checkpoints/; bash ~/data/megatron-lm/megatron/native/scripts/enroot.sh run |&  tee /tmp/out.log |& grep -E 'native_ckpt.*perf' > ~/data/megatron-lm/megatron/native/perf-rslt-cwdfw.log &

```
Notes for a 32GB buffer

Pinned buffer registration time : ~9s on first attempt for 32GB buffer and 3s on repeate
+0: [native_ckpt:cli][perf] Time=9509999us Registering host buffer with CUDA. addr=0x1543effff000 size=34359742464
+1: [native_ckpt:cli][perf] Time=9642929us Registering host buffer with CUDA. addr=0x15448ffff000 size=34359742464
+0: [native_ckpt:cli][perf] Time=2669163us Registering host buffer with CUDA. addr=0x1543effff000 size=34359742464
+1: [native_ckpt:cli][perf] Time=2826603us Registering host buffer with CUDA. addr=0x15448ffff000 size=34359742464


+0: [native_ckpt:cli][0][INFO][perf] Finished Time: DeviceInit=9510044us StateSer=53us BcastMeta=677us WrTnsr=164345us TotalTime=9675121us AvgD2HRq=497039361 NumTensor=1063 AvgTnsr=7481307 SmallD2H=2 TotalD2H=16 TotalWritten=7952629776
+1: [native_ckpt:cli][1][INFO][perf] Finished Time: DeviceInit=9642962us StateSer=13221us BcastMeta=19909us WrTnsr=170912us TotalTime=9847005us AvgD2HRq=497039361 NumTensor=1063 AvgTnsr=7481307 SmallD2H=2 TotalD2H=16 TotalWritten=7952629776
+0: [native_ckpt:cli][0][INFO][perf] Finished Time: DeviceInit=2669188us StateSer=54us BcastMeta=364us WrTnsr=164107us TotalTime=2833715us AvgD2HRq=497039361 NumTensor=1063 AvgTnsr=7481307 SmallD2H=2 TotalD2H=16 TotalWritten=7952629776
+1: [native_ckpt:cli][1][INFO][perf] Finished Time: DeviceInit=2826624us StateSer=7039us BcastMeta=291us WrTnsr=170723us TotalTime=3004678us AvgD2HRq=497039361 NumTensor=1063 AvgTnsr=7481307 SmallD2H=2 TotalD2H=16 TotalWritten=7952629776
```

# devbox

rm -rf ~/megatron-lm/runs/checkpoints/; bash ~/megatron-lm/megatron/native/scripts/run.sh |& tee /tmp/out.log |& grep -E 'native_ckpt.*perf' > ~/megatron-lm/megatron/native/perf-rslt-devbox.log &
