import heapq,csv
from collections import defaultdict
edges={1:[(2,10),(4,10)],2:[(1,10),(3,20),(5,10)],3:[(2,20),(7,10)],4:[(1,10),(5,10),(8,10),(6,20)],5:[(2,10),(4,10),(6,10),(9,10)],6:[(5,10),(7,10),(4,20)],7:[(3,10),(6,10),(10,10)],8:[(4,10),(9,10),(11,10)],9:[(5,10),(8,10),(10,20),(12,10),(13,15)],10:[(7,10),(9,20),(13,15),(17,20)],11:[(8,10),(12,10),(14,10)],12:[(9,10),(11,10),(15,10)],13:[(9,15),(10,15),(16,10)],14:[(11,10),(15,10),(16,20)],15:[(12,10),(14,10),(16,10)],16:[(13,10),(15,10),(17,10),(14,20)],17:[(10,20),(16,10)]}
stations={'cycle':{1,2,3,5,7,9,10,12,13,15,17},'car':{2,3,5,7,9,10,12,13,15,17},'bus':{2,3,5,7,9,10,12,13,15,17},'train':{4,6,8,11,14,16,17}}
speeds={'cycle':20,'car':100,'bus':80,'train':90}
# Train-only track restriction in uploaded PRISM model.
train_only={(4,6),(6,4),(14,16),(16,14)}
train_block={(4,1),(4,5),(6,5),(6,7),(8,9),(11,12),(14,15),(16,13),(16,15)}

def solve(p,lanes=0,correct_transfer=False):
  pq=[(0,1,'none')]; dist={(1,'none'):0}; prev={};
  while pq:
    cost,u,mode=heapq.heappop(pq)
    if cost>dist[(u,mode)]+1e-9:continue
    if u==17 and (mode=='car' or u in stations[mode]):
      route=[];k=(u,mode)
      while k in prev:
        prior,step=prev[k];route.append(step);k=prior
      return cost,list(reversed(route))
    possible=[mode] if mode!='none' else []
    if mode=='none' or mode=='car' or u in stations[mode]:
      possible+= [m for m in speeds if m!=mode and u in stations[m]]
    for m in possible:
      for v,length in edges[u]:
        if (u,v) in train_only and m!='train':continue
        if (u,v) in train_block and m=='train':continue
        # destination rewards after traversing, as in calculate command
        pc=0 if m in ('train','cycle') or (m=='bus' and lanes) else p
        travel=60*length/max((1-pc)*speeds[m],0.0001)
        transfer=(5 if correct_transfer else 5/60) if m!=mode else 0
        new=cost+travel+transfer
        key=(v,m)
        if new<dist.get(key,float('inf'))-1e-9:
          dist[key]=new;prev[key]=((u,mode),(u,v,m,round(travel+transfer,5)));heapq.heappush(pq,(new,v,m))
  return float('inf'),[]
if __name__=='__main__':
  for lanes in [0,1]:
    for p in [0,.2,.4,.6,.8,1.0]:
      val,route=solve(p,lanes)
      print(lanes,p,round(val,5),[(a,b,m) for a,b,m,_ in route])
