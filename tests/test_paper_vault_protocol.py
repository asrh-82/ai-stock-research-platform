"""Execute the real component script against a deterministic browser-protocol stub."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_acknowledgments_do_not_emit_a_second_read_or_overwrite_conflicts():
    js = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8');
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
let text=null, denied=false;
const messages=[];
const context=vm.createContext({
  window:{parent:{postMessage:m=>messages.push(m)},addEventListener:()=>{}},
  document:{getElementById:()=>({textContent:''})},
  navigator:{locks:{request:async (name,fn)=>await fn()}},
  localStorage:{getItem:()=>text,setItem:(k,v)=>{if(denied)throw Error('quota');text=v;}},
  TextEncoder, console
});
vm.runInContext(source,context);
const values=()=>messages.filter(m=>m.type==='streamlit:setComponentValue').map(m=>m.value);
async function render(command){
  context.args={command};
  await vm.runInContext('render(args)',context);
}
(async()=>{
  await render(null); assert.equal(values().length,1);
  const doc={schema:1,revision:1,events:[{id:'example'}]};
  await render({id:'first',base_revision:0,document:doc});
  assert.equal(values().at(-1).ack,'first');
  const count=values().length;
  await render(null);
  assert.equal(values().length,count,'No duplicate read may interrupt the next UI action');
  await render({id:'first',base_revision:0,document:doc});
  assert.equal(values().at(-1).ack,'first','Write retry is idempotent');
  await render({id:'conflict',base_revision:0,document:{...doc,revision:2}});
  assert(values().at(-1).error.includes('Another tab'));
  assert.equal(JSON.parse(text).document.revision,1);
  denied=true;
  await render({id:'blocked',base_revision:1,document:{...doc,revision:2}});
  assert.equal(values().at(-1).ready,false);
  assert.equal(JSON.parse(text).document.revision,1,'Failed write must retain prior record');
  console.log('VAULT_PROTOCOL_PASSED');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(['node', '-e', js, str(ROOT/'components/paper_vault/index.html')],
                            text=True, capture_output=True, check=True)
    assert 'VAULT_PROTOCOL_PASSED' in result.stdout
