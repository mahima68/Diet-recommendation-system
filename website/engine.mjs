const activity = ['Sedentary','Lightly Active','Moderately Active','Very Active','Extremely Active'];
const meals = ['Breakfast','Lunch','Dinner','Snack'];
export function conflict(text,pref){
 text=text.toLowerCase().replace(/\b(?:vegan|plant-based|dairy-free|non-dairy)\s+(?:yogurt|yoghurt|cheese|milk|butter|cream)\b/g,'').replace(/\b(?:almond|soy|soya|oat|coconut|peanut|cashew|sunflower)\s+(?:milk|butter|yogurt|cream)\b/g,'');
 return (pref!=='Omnivore' && /\b(chicken|beef|pork|steak|turkey|bacon|ham|lamb|sausage)\b/.test(text)) || (['Vegan','Vegetarian'].includes(pref) && /\b(fish|salmon|tuna|shrimp|prawn|cod|trout|sardine|anchovy|seafood)\b/.test(text)) || (pref==='Vegan' && /\b(egg|eggs|yogurt|yoghurt|cheese|milk|butter|cream|honey)\b/.test(text));
}
export function recommend(model,p){
 if(!Number.isFinite(p.age)||p.age<18||p.age>100||!Number.isFinite(p.weight)||p.weight<30||p.weight>300||!activity.includes(p.activity)||!['Omnivore','Pescatarian','Vegan','Vegetarian'].includes(p.preference)) throw Error('Please check your profile values.');
 if(p.calories!=null && (!Number.isFinite(p.calories)||p.calories<1||p.calories>10000))throw Error('Calorie target must be between 1 and 10,000.');
 const labels=model.features.slice(3), conditions=p.conditions||[];
 if(conditions.some(c=>!labels.includes(c)))throw Error('Unknown dataset label.');
 if(p.tolerance!=null && (p.calories==null||!Number.isFinite(p.tolerance)||p.tolerance<0||p.tolerance>1))throw Error('Choose a calorie target before setting a tolerance.');
 const vector=[p.age,p.weight,activity.indexOf(p.activity),...labels.map(c=>+conditions.includes(c))].map((x,i)=>(x-model.mean[i])/model.scale[i]);
 const norm=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0)), vn=norm(vector);
 const candidates=model.data.flatMap((row,index)=>{
  if(row['Dietary Preference']!==p.preference || conflict(meals.map(m=>row[m+' Suggestion']).join(' '),p.preference))return [];
  const diff=p.calories==null?0:Math.abs(row.Calories-p.calories);
  if(p.tolerance!=null&&diff>p.calories*p.tolerance)return [];
  const diseases=row.Disease.split(',').map(x=>x.trim());
  const raw=[row.Ages,row.Weight,activity.indexOf(row['Activity Level']),...labels.map(c=>+diseases.includes(c))].map((x,i)=>(x-model.mean[i])/model.scale[i]);
  const den=norm(raw)*vn, similarity=den?raw.reduce((s,x,i)=>s+x*vector[i],0)/den:0;
  let score=(similarity+1)/2;
  if(p.calories!=null)score=.7*score+.3*Math.max(0,Math.min(1,1-diff/p.calories));
  return [{...row,score,index}];
 }).sort((a,b)=>b.score-a.score||a.index-b.index);
 const seen=new Set();return candidates.filter(r=>{const k=JSON.stringify(meals.map(m=>r[m+' Suggestion']));if(seen.has(k))return false;seen.add(k);return true;}).slice(0,5);
}
