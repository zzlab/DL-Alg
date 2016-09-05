'''
GPU_availability returns [least_occupied_GPU, ..., most_occupied_GPU].
Each element of the list is an GPU index (GPU index starts from 0).
It is ensured that the current performance of each GPU in the list is at most P2.
P0 is the maximum performance, indicating that one GPU is completely occupied.
P12 is the minimum performance.
'''

def GPU_availability():
  import itertools
  import re
  from subprocess import Popen, PIPE
  output = Popen(['nvidia-smi'], stdout=PIPE).communicate()[0]
  lines = output.split('\n')
  performance = {}
  memory = {}
  index = 0
  for i in range(len(lines)):
    if 'GTX' in lines[i]:
      line = lines[i+1]
      p = int(re.search('P\d', line).group(0)[-1])
      if p>-1:
        match = re.findall('(\d+)MiB', line)
        available = int(match[1]) - int(match[0])
        memory.update({index : available})
        try:
          performance[p].append(index)
        except:
          performance.update({p : [index]})
      index += 1
  return list(itertools.chain(*[
    sorted(performance[key], cmp=lambda l,r: cmp(memory[l], memory[r]), reverse=True) \
    for key in reversed(sorted(performance.keys()))]))

def GPU_memory(index):
  import re
  from subprocess import Popen, PIPE
  output = Popen(['nvidia-smi'], stdout=PIPE).communicate()[0]
  lines = output.split('\n')
  current = 0
  for i in range(len(lines)):
    if 'GTX' in lines[i]:
      if current == index:
        line = lines[i+1]
        match = re.findall('(\d+)MiB', line)
        available = int(match[1]) - int(match[0])
        return available, int(match[0]), int(match[1])
      current += 1
