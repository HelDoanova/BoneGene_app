from re import I
import dolfin as df
from pandas import value_counts
import panels
import pyvista as pv
import numpy as np
import panel as pn
import bokeh
import os
import logging
import time
import vtk
import io
import uuid
from pyDOE import lhs
from scipy import stats
import panel.io.loading as li
import zmq
import copy
from bokeh.io import curdoc

curdoc().clear()


css = '''
.navbar{
    font-size: 20px;
}

.bk-root .bk{
    font-size: 16px;
}

.bk.pn-loading.arcs::before{  
    background-size: 180px 180px;
    max-height: 100%;
    max-width: 100%;
    background-position: center;
    content: "Loading data ...";
    color: #4a4a4a;
    text-align: center;
    padding-top: 36%;
    font-size: 38px;
}

.bk-root .bk-btn-default{
    padding: 10px 0;
    font-size: 14px;
}
'''

pn.extension('vega', 'katex', 'mathjax', 'vtk', sizing_mode='stretch_width', raw_css=[css])
bootstrap = pn.template.BootstrapTemplate(title='BoneGen', header_background='#ee7f00', sidebar_width=500)


LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
file_handler = logging.FileHandler(filename='test.log', mode='w')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# TEXT ----------------------------------------------------------------------------------------------------------------------------------------
text_sex = pn.pane.Markdown(""" <br> Note that BMD is different for female and male. Therefore, **sex** must be selected using the two buttons below: """)
text_age = pn.pane.Markdown(""" <br> The **age** significantly influences the BMD. BMD generally leads to lower values with age. 
The age range below is given by the dataset used in the cited study. """)
text_accuracy = pn.pane.Markdown(""" <br> The **accuracy** of random field representation is controlled by a number of eigenpairs included in KL expansion. 
The mode #1 is the most significant, and the contribution is the minimum for the last KL term (the lowest eigenvalue). 
t is not recommended use all eigenpairs because of web application limited performance. By default, 
five eigenpairs are used and truncation can be controlled by slider below: """)
text_realisation = pn.pane.Markdown(""" <br> To compute random **realisation** of BMD, the coeficients θ must be generated with LHS design. 
Choose how many realisations should be generated at once. """)
text_bone_select = pn.pane.Markdown(""" <br> Select the **bone types** you want to render. There is no limit for selection.""", height=10)
endgap_box1 = pn.pane.Markdown(""" <br> """, height = 10)
endgap_box2 = pn.pane.Markdown(""" <br> """, height = 10)
endgap_box3 = pn.pane.Markdown(""" <br> """, height = 10)
btn_load_line_up = pn.pane.Markdown(""" ------------""", height=20)                                           
load_down_gap = pn.pane.Markdown(""" <br> """, height = 10)
paraview_txt = pn.pane.Markdown(""" Once the realisation is available, can be saved as *.vtu file, which can be opened with [Paraview](https://www.paraview.org/).""", height=50)
info_line = pn.pane.Markdown(""" ------------""", height = 20)
text_warning = pn.pane.Markdown(""" The web application was developed by Helena Doanová and serves as demonstration of developed computational method in the [study](https://www.biorxiv.org/content/10.1101/2021.02.25.432881v1). 
Using results from this web application follows the CC licence. The web application should not be used for medical diagnosis. For any further information, send <a href = "mailto: doanova.hell@gmail.com">email</a> to the authors.""", height=0)

# CLASS -------------------------------------------------------------------------------------------------------------------------------------
class KLE(object):
    #bone_names = ["femur_left","femur_right","ilium_left","ilium_right","sacrum","l5"]  #datasets for bones
    #bone_type =    ["Ilium R", "Ilium L", "Femur R", "Femur L", "Sacrum", "L5"]
    #bone_type =["Ilium R", "Ilium L", "Femur R"]

    def __init__(self, plotter_surface):
        self.plotter_surface = plotter_surface

        self.bone_type =["Ilium R", "Ilium L", "Femur R"]
        #self.bone_type =["Ilium R"]
        self.data_loaded = False
        self.vtkpan = None
        self.download = None
        self.weight_list = []

        self.meshes_rainbow = {}
        self.meshes_greys = {}
        self.actor_collection_rainbow = vtk.vtkActorCollection()
        self.actor_collection_greys = vtk.vtkActorCollection()


        # self.renderer = list(self.plotter_surface.ren_win.GetRenderers())[0]
        # self.initial_camera = self.renderer.GetActiveCamera()
        # # self.initial_camera_pos = {"focalPoint": self.initial_camera.GetFocalPoint(),
        # #                             "position": self.initial_camera.GetPosition(),
        # #                             "viewUp": self.initial_camera.GetViewUp()}
        #self.initial_camera_pos = {}
        self.bmd_fun = None
        #self.filename = 'bmd000000.vtu'


    def load_meshes(self):
        logger.info('Loading data in KLE')
        for bone_type in self.bone_type:
            self.meshes_rainbow[bone_type] = pv.read(bone_type + '/average_patient.xml') 

        self.meshes_greys = copy.deepcopy(self.meshes_rainbow)
        self.data_loaded = True
        logger.info('Data loaded in KLE')

    def add_meshes_to_plotter_surface(self):
        li.start_loading_spinner(model.vtkpan)
        logger.info('Adding meshes to plotter_surface in KLE')
        for i, bone_type in enumerate(self.bone_type):
            KLE.update_meshes(self, self.meshes_rainbow[bone_type]) 

            mesh_actor_rainbow=self.plotter_surface.add_mesh(self.meshes_rainbow[bone_type].translate(100*i,100,100), cmap='rainbow',show_scalar_bar=True, scalar_bar_args={'title': 'BMD [g/cc]'})     #pridavani dat do plottru i s posunem jednotlivých kostí pomocí translatte()
            mesh_actor_rainbow.VisibilityOff()                  #aby se nam zobrazovali jen když my chceme
            self.actor_collection_rainbow.AddItem(mesh_actor_rainbow)           

            mesh_actor_greys=self.plotter_surface.add_mesh(self.meshes_greys[bone_type].translate(100*i,100,100), cmap='Greys', show_scalar_bar=True, scalar_bar_args={'title': 'BMD [g/cc]'})     #pridavani dat do plottru i s posunem jednotlivých kostí pomocí translatte()
            mesh_actor_greys.VisibilityOff()                  #aby se nam zobrazovali jen když my chceme
            self.actor_collection_greys.AddItem(mesh_actor_greys)

            logger.info('Mesh added with translate')
        logger.info('Meshes added in KLE')
        li.stop_loading_spinner(model.vtkpan)
        
    def update_meshes(slef, bone):
        init_data = np.zeros(bone.n_cells) + 0.1    
        init_data[-1] = 1.2                                           
        bone.cell_data['BMD [g/cc]'] = init_data  

    def create_vtkpanel(self):
        self.vtkpan = pn.panel(self.plotter_surface.ren_win, orientation_widget=True, interactive_orientation_widget=True, enable_keybindings=True, width=1450, height=850)
        bootstrap.main.append(pn.Row(pn.layout.HSpacer(),self.vtkpan, pn.layout.HSpacer()))


# COMPONENTS (widget, pane, panels) ----------------------------------------------------------------------------------------------------------
# SIDEBAR
#sex_selection
sex_buttons = pn.widgets.RadioButtonGroup(
    name='Sex selection', options=['woman', 'man'], button_type='warning')

#age_selection
ut_men = 88         # co znamená tato zkratka ?
ut_women = 97       #   
age_slider = pn.widgets.IntSlider(name='age, t[year]', start=22, end=89, step=1, value=55)

#bones_selection                                #prejmenovat kosti
bone_chcecksbox = pn.widgets.CheckBoxGroup(name='Bone selection', value=['Ilium R'], 
                                            options=['Ilium R', 'Ilium L', 'Femur R', 'Femur L', 'Sacrum','L5'], 
                                            sizing_mode=None, width=100)
jpg_pelvic_bone = pn.pane.JPG('/home/hela/Desktop/BoneGen/img/panev.jpg', width=330)
#<a href="https://www.freepik.com/vectors/pelvis">Pelvis vector created by brgfx - www.freepik.com</a>
above_chceckBox_gap = pn.pane.Markdown(""" <br> """, height = 120, width=100)
chceckBox_gap=pn.Column(above_chceckBox_gap, bone_chcecksbox)
above_jpg_gap = pn.pane.Markdown(""" <br> """, height=30)           ###
jpg_gap = pn.Column(above_jpg_gap, jpg_pelvic_bone) ###

#accuracy_selection
KL_slider =  pn.widgets.IntRangeSlider(name='KL range',start=1, end=97, value=(1, 5), step=1)

#selection_table
box1 = pn.WidgetBox(text_sex, sex_buttons,text_age, age_slider, endgap_box1) #height=400
#box2 = pn.WidgetBox(text_bone_select,pn.Row(chceckBox_gap, jpg_pelvic_bone) ,endgap_box2)
box2 = pn.WidgetBox(text_bone_select, pn.Row(chceckBox_gap, jpg_gap), endgap_box2)
box3 = pn.WidgetBox(text_accuracy, KL_slider, endgap_box3)

tabs = pn.Tabs(('TAB1',box1), ('TAB2',box2), ('TAB3',box3))

#button_draw_realisation
btn_start_simulation = pn.widgets.Button(name='Start simulation', button_type='success', disabled=True)

def get_bmd_fun():
    for bone_type in bone_chcecksbox.value:
        mesh_fenics = df.Mesh(bone_type + '/average_patient.xml') #POZOR TOTO VYRESIT/ZEPTAT SE PROFESORA
        V = df.FunctionSpace(mesh_fenics, 'DG', 0)
        model.bmd_fun = df.Function(V, name='BMD')

def save_realisation():
    li.start_loading_spinner(model.vtkpan)
    for bone_type in bone_chcecksbox.value:
        current = model.meshes_rainbow[bone_type]
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(model.download)  ###
    writer.SetInputData(current) # mesh should be a PolyData object
    writer.Write()
    writer.SetDataModeToBinary()
    writer.WriteToOutputStringOn()
    writer.Update()
    xml = writer.GetOutputString()
    sio = io.StringIO(xml)
    sio.seek(0)
    li.stop_loading_spinner(model.vtkpan)
    logger.info('Saving file end')
    return sio

    
file_download = pn.widgets.FileDownload(callback=save_realisation, filename='BMD_realisation.vtu', height = 38)

def add_componets_to_sidebar():
    bootstrap.sidebar.append(tabs)
    bootstrap.sidebar.append(btn_start_simulation)
    bootstrap.sidebar.append(btn_load_line_up)
    bootstrap.sidebar.append(load_down_gap)
    bootstrap.sidebar.append(paraview_txt)
    bootstrap.sidebar.append(file_download)
    bootstrap.sidebar.append(info_line)
    bootstrap.sidebar.append(text_warning)

add_componets_to_sidebar()

# MAIN--------------------------------------------------------------------------------------------------------------
#pyvista plotter
def create_model():  
    plotter_surface = pv.Plotter(notebook=True) 
    plotter_surface.background_color = (0.6, 0.6, 0.6)

    model = KLE(plotter_surface)
    model.create_vtkpanel()
    return model

model = create_model()

select_cmap = pn.widgets.Select(options=['rainbow', 'Greys'],sizing_mode=None, width=250)
cmap_change = pn.widgets.Button(name='Change cmap', button_type='primary', sizing_mode=None, width=250)

keybindings= pn.Row(
    pn.Column(
        pn.pane.Markdown(""" - set representation of all actors to surface:     **S**""", height = 15, width=340),
        pn.pane.Markdown(""" - set representation of all actors to wireframe:   **W**""", height = 15, width=365)
    ),
    pn.Column(
        pn.pane.Markdown(""" - center the actors and reset camera:    **R**""",height = 15, width=360),
        pn.pane.Markdown(""" - set representation of all actors to vertex:      **V**""")
        
    ), sizing_mode=None, width=750
)

# DEF--------------------------------------------------------------------------------------------------------------
                        
def draw_realisation(event):
    li.start_loading_spinner(model.vtkpan)
    btn_start_simulation.disabled = True
    set_visibility_actor()
    logger.info('loading from server')  
    print(model.plotter_surface) 
    get_bmd_fun()
    for bone_type in bone_chcecksbox.value:
        dd = dict(
        sex = sex_buttons.value,
        bone_type = bone_type,
        age = age_slider.value,
        spb = KL_slider.value
        )
        bmd = recv_array(dd)
        model.meshes_rainbow[bone_type].cell_data['BMD [g/cc]'] = bmd
        model.meshes_rainbow[bone_type].set_active_scalars('BMD [g/cc]')
        model.bmd_fun.vector().set_local(bmd)
        model.weight_list.append((df.assemble(model.bmd_fun * df.dx) * 1e-3, age_slider.value))  #spocitani modelu kosti
        np.save("meshes_weight_age", model.weight_list)
        print(model.weight_list[-1])
        print("----------")
    
    logger.info('vtkpan start synchronized') 
    model.vtkpan.reset_camera()
    model.vtkpan.synchronize()
    
    li.stop_loading_spinner(model.vtkpan)
    logger.info('vtkpan end synchronized')
    btn_start_simulation.disabled = False
    file_download.disabled=False


# def draw_realisation(event):
#     my_age = 12
#     for i in range(7):
#         my_age = my_age + 10
#         for i in range(120):
#             li.start_loading_spinner(model.vtkpan)
#             btn_start_simulation.disabled = True
#             set_visibility_actor()
#             logger.info('loading from server')  
#             #print(model.plotter_surface) 
#             get_bmd_fun()
#             for bone_type in bone_chcecksbox.value:
#                 dd = dict(
#                 sex = sex_buttons.value,
#                 bone_type = bone_type,
#                 age = my_age,  #age_slider.value,
#                 spb = KL_slider.value
#                 )
#                 bmd = recv_array(dd)
#                 model.meshes_rainbow[bone_type].cell_data['BMD [g/cc]'] = bmd
#                 model.meshes_rainbow[bone_type].set_active_scalars('BMD [g/cc]')
#                 model.bmd_fun.vector().set_local(bmd)
#                 model.weight_list.append((df.assemble(model.bmd_fun * df.dx) * 1e-3,my_age)) #age_slider.value))  #spocitani modelu kosti
#                 np.save("meshes_weight_age", model.weight_list)
#                 print(model.weight_list[-1])
#                 print(i)
#                 print("----------")
            
#             logger.info('vtkpan start synchronized') 
#             #print(model.plotter_surface.scalar_bars)
#             # model.vtkpan.reset_camera()
#             # model.vtkpan.synchronize()
#             li.stop_loading_spinner(model.vtkpan)
#             #logger.info('vtkpan end synchronized')
#             btn_start_simulation.disabled = False
#             file_download.disabled=False
#             #btn_draw_realisation.disabled = True
    


btn_start_simulation.on_click(draw_realisation)

def recv_array(model_parameters, flags=0, copy=True, track=False):
        context = zmq.Context()
        #  Socket to talk to server
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        """recv a numpy array"""
        socket.send_json(model_parameters)

        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])

        return A.reshape(md['shape'])

def set_visibility_actor():
    cmap=select_cmap.value
    if cmap == "rainbow":
        collection_on = model.actor_collection_rainbow
        collection_off = model.actor_collection_greys
    elif cmap == "Greys":
        collection_on = model.actor_collection_greys
        collection_off = model.actor_collection_rainbow

#VISIBILITY
    for i, bone_type in enumerate(model.bone_type):
        collection_on.GetItemAsObject(i).VisibilityOn()
        collection_off.GetItemAsObject(i).VisibilityOff()
        if bone_type not in bone_chcecksbox.value:
            bone_index = model.bone_type.index(bone_type)
            select_actor = collection_on.GetItemAsObject(bone_index)
            select_actor.VisibilityOff()

def load_meshes_action(event=None):
    logger.info("Load_meshes start")
    li.start_loading_spinner(model.vtkpan)
    model.load_meshes()
    
    li.stop_loading_spinner(model.vtkpan)
    logger.info("Button load data end and spinner")
    logger.info("Load_meshes end")

def update_cmap(event):
    li.start_loading_spinner(model.vtkpan)
    set_visibility_actor()
    print("synch in update cmap")
    model.vtkpan.synchronize()
    li.stop_loading_spinner(model.vtkpan)

cmap_change.on_click(update_cmap)


@pn.depends(value=sex_buttons.param.value)                          
def adjust_KL_slider(value):
    if value=="woman":
        KL_slider.param.set_param(start=1, end=ut_women, value=(1, 5))
    elif value=="man": 
        KL_slider.param.set_param(start=1, end=ut_men, value=(1, 5))

def gen_file_name(age, sex, bone, KLn, cutout=8):
    prefix = 'BMD-{}-{}-{}-{}'.format(age, sex, bone, KLn)
    hash =  str(uuid.uuid4())[:cutout] 
    #model.filename = 'BMD{}{}{}'.format(age, sex, KLn) + hash + '.vtu'
    model.download = prefix + hash + '.vtu'
    return model.download

@pn.depends(age_slider.param.value, KL_slider.param.value, 
            sex_buttons.param.value,bone_chcecksbox.param.value, btn_start_simulation.param.clicks)
def update_fname(age, KLn, sex, bone, sbclick):
    lt, ut = KLn
    fname = gen_file_name(age, sex, bone, ut - lt + 1)
    file_download.filename = fname
    file_download._update_filename()
    file_download.disabled=True

@pn.depends(btn_start_simulation.param.clicks)
def activate_save(clicks):
    if clicks==0:
        file_download.disabled=True
    else:
        file_download.disabled = False

def add_component_to_main():
    bootstrap.main.append(pn.Row(pn.layout.HSpacer(),keybindings, select_cmap, cmap_change,pn.layout.HSpacer()))
    bootstrap.sidebar.append(adjust_KL_slider)
    bootstrap.sidebar.append(update_fname)
    bootstrap.sidebar.append(activate_save)

add_component_to_main()

#--------------------------------------------------------------------------------------------------------

def app():
    con = bootstrap.servable();
    li.start_loading_spinner(model.vtkpan)

    def on_load():
        time.sleep(3)
        print("spustim peload data")
        load_meshes_action()
        model.add_meshes_to_plotter_surface()
        print("konec preload data")
        btn_start_simulation.disabled = False

    pn.state.onload(on_load)

    return con

pn.serve(app) 

#bootstrap.servable();

#http://localhost:5006/Bonegen_v1


