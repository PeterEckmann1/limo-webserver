from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import sqlite3
from secrets import token_hex
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import File, UploadFile
import subprocess
import os


binding_center_script = '''
from pymol import cmd


cmd.load('{}')
cmd.pseudoatom('binding_center', pos=[{}, {}, {}])
cmd.iterate('all within 7.5 of binding_center', 'print resi')'''


app = FastAPI()
app.mount('/imgs', StaticFiles(directory='imgs'), name='imgs')
conn = sqlite3.connect('/data/peter/mol.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

origins = ['*']

standard_res = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'PYL', 'SEC', 'DAL', 'DSN', 'DCY', 'DPR', 'DVA', 'DTH', 'DLE', 'DIL', 'DSG', 'DAS', 'MED', 'DGN', 'DGL', 'DLY', 'DHI', 'DPN', 'DAR', 'DTY', 'DTR', 'DA', 'DC', 'DG', 'DT', 'DI', 'A', 'C', 'G', 'U', 'I', 'ASX', 'GLX', 'UNK']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/get_mols')
async def get_mols(target, qed_slider, sa_slider, nitrogen_slider, oxygen_slider):
    #todo record different slider values
    out = []
    for row in cur.execute('select smiles, autodock_affin, amber_affin, qed, sa, logp, docking_rmsd, dock_pose from mols where target=?', (target,)).fetchall():
        out.append(dict(row))
    qed_cutoff = np.quantile([mol['qed'] for mol in out], float(qed_slider))
    sa_cutoff = np.quantile([mol['sa'] for mol in out], float(sa_slider))
    nitrogen_lower_cutoff = np.quantile([(mol['smiles'].count('N') + mol['smiles'].count('n')) for mol in out], min(0.9, float(nitrogen_slider)))
    nitrogen_upper_cutoff = np.quantile([(mol['smiles'].count('N') + mol['smiles'].count('n')) for mol in out], min(1.0, float(nitrogen_slider) + 0.1))
    oxygen_lower_cutoff = np.quantile([(mol['smiles'].count('O') + mol['smiles'].count('o')) for mol in out], min(0.9, float(oxygen_slider)))
    oxygen_upper_cutoff = np.quantile([(mol['smiles'].count('O') + mol['smiles'].count('o')) for mol in out], min(1.0, float(oxygen_slider) + 0.1))
    res = []
    res = []
    for mol in out:
        if mol['qed'] >= qed_cutoff and mol['sa'] <= sa_cutoff and nitrogen_lower_cutoff <= (mol['smiles'].count('N') + mol['smiles'].count('n')) <= nitrogen_upper_cutoff and oxygen_lower_cutoff <= (mol['smiles'].count('O') + mol['smiles'].count('o')) <= oxygen_upper_cutoff:
            filename = 'imgs/' + mol['smiles'].encode('utf-8').hex() + '.svg'
            MolToFile(MolFromSmiles(mol['smiles']), filename)
            mol['img'] = filename
            res.append(mol)
    return {'mols': res}


@app.post('/upload_from_file')
async def upload_from_file(file: UploadFile = File(...)):
    target_id = token_hex(8)
    f_name = f'/data/peter/pdbs/{target_id}.pdb'
    if not file.filename.endswith('.pdb'):
        return {'error': 'invalid pdb'}
    contents = file.file.read()
    if len(contents) > 10000000:
        return {'error': 'file too large'}
    open(f_name, 'wb').write(contents)
    subprocess.run(f'obabel {f_name} -O {f_name}', shell=True)
    hets = []
    for line in open(f_name, 'r'):
        if line.startswith('HET '):
            hets.append(f'{line.split()[1]}:{line.split()[2]}')
    return {'target_id': target_id, 'nonstandard_res': list(set(hets))}


@app.get('/get_binding_site')
async def get_binding_site(target_id, res=None, x=None, y=None, z=None):
    f_name = f'/data/peter/pdbs/{target_id}.pdb'
    ligand_name = f'/data/peter/pdbs/{target_id}_ligand.pdb'
    os.chdir('/data/peter/pdbs')
    subprocess.call(f'python2 ~/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {f_name}', shell=True)
    if res:
        res_lines = []
        for line in open(f_name, 'r'):
            if line.startswith('HETATM') and res.replace(':', ' ') in line:
                res_lines.append(line.strip())
        open(ligand_name, 'w').write('\n'.join(res_lines))
        subprocess.run(f'obabel {ligand_name} -O {ligand_name}qt', shell=True)
        subprocess.call(f'python2 ~/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_gpf4.py -l {ligand_name}qt -r {f_name}qt -y', shell=True)
        x, y, z = map(float, open(f_name.replace('.pdb', '.gpf'), 'r').read().split('gridcenter')[1].split('\n')[0].split()[:3])
    open('binding_center.py', 'w').write(binding_center_script.format(target_id + '.pdb', x, y, z))
    out = subprocess.run('pymol -cq binding_center.py', shell=True, capture_output=True)
    res_nums = map(int, out.stdout.decode('utf-8').strip().split('\n'))
    return {'x': x, 'y': y, 'z': z, 'residue_nums': list(set(res_nums))}


@app.post('/submit_job')
async def submit_job(target_id, x, y, z, email):
    return {'status': 'success'}


@app.get('/get_structure')
async def get_structure(smiles, target):
    os.chdir('/home/peter/webapp')
    structure = list(cur.execute('select autodock_pose from mols where smiles=? and target=?', (smiles, target)).fetchone())[0]
    structure += open(f'../limo/proteins/{target}/{target}.pdb', 'rb').read()
    structure = structure.decode('utf-8').replace('ENDMDL', '').encode('utf-8')
    return PlainTextResponse(structure, media_type='chemical/x-pdb', headers={'Content-Disposition': 'attachment; filename=structure.pdb'})


@app.get('/record_questionnaire')
async def record_questionnaire(question1, question2, question3):
    cur.execute('insert into question_responses values (?, ?, ?)', (question1, question2, question3))
    conn.commit()
    return {'status': 'success'}


@app.get('/record_thumbs')
async def record_thumbs(target, smiles, response):
    cur.execute('update mols set thumbs=? where target=? and smiles=?', (int(response), target, smiles))
    conn.commit()
    return {'status': 'success'} 


@app.get('/record_comment')
async def record_comment(target, smiles, response):
    cur.execute('update mols set comment=? where target=? and smiles=?', (response, target, smiles))
    conn.commit()
    return {'status': 'success'}