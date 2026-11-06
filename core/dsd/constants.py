"""Shared constants. Tooth order is left -> right as seen in a frontal photo
(the patient's right side appears on the viewer's left)."""

TEETH = ['13', '12', '11', '21', '22', '23']
CLASS_OF = {'13': 1, '12': 2, '11': 3, '21': 4, '22': 5, '23': 6}
TOOTH_NAME = {'13': 'ناب أيمن', '12': 'رباعية يمنى', '11': 'ثنية يمنى',
              '21': 'ثنية يسرى', '22': 'رباعية يسرى', '23': 'ناب أيسر'}

PROPORTIONS = {
    'golden': {'label': 'النسبة الذهبية (62%)', 'p': 0.618},
    'red70': {'label': 'RED ثابتة 70%', 'p': 0.70},
    'red75': {'label': 'RED ثابتة 75%', 'p': 0.75},
    'red80': {'label': 'RED ثابتة 80%', 'p': 0.80},
}

# Literature-based starting values; the doctor edits them. Same ids and values as the JavaScript app.
DEFAULT_RULES = [
    dict(id='central_wl', group='teeth', label='نسبة عرض الثنية إلى طولها', min=0.75, max=0.85, target=0.80, unit='ratio', enabled=True),
    dict(id='lat_central_w', group='teeth', label='عرض الرباعية إلى الثنية (ظاهري)', tol=0.06, unit='ratio', enabled=True),
    dict(id='can_lat_w', group='teeth', label='عرض الناب إلى الرباعية (ظاهري)', tol=0.08, unit='ratio', enabled=True),
    dict(id='lat_central_len', group='teeth', label='طول الرباعية إلى طول الثنية', min=0.80, max=0.90, target=0.85, unit='ratio', enabled=True),
    dict(id='can_central_len', group='teeth', label='طول الناب إلى طول الثنية', min=0.88, max=1.00, target=0.95, unit='ratio', enabled=True),
    dict(id='sym_central', group='teeth', label='تماثل الثنيتين (فرق العرض والطول)', max=0.05, unit='pct', enabled=True),
    dict(id='sym_lateral', group='teeth', label='تماثل الرباعيتين', max=0.08, unit='pct', enabled=True),
    dict(id='sym_canine', group='teeth', label='تماثل النابين', max=0.10, unit='pct', enabled=True),
    dict(id='central_dominance', group='teeth', label='سيادة الثنيتين (أعرض الأسنان ظاهرياً)', unit='bool', enabled=True),
    dict(id='central_len_mm', group='teeth', label='طول الثنية بالملم', min=10.0, max=11.5, target=10.8, unit='mm', needs='calib', enabled=True),
    dict(id='central_w_mm', group='teeth', label='عرض الثنية بالملم', min=8.0, max=9.5, target=8.6, unit='mm', needs='calib', enabled=True),
    dict(id='zenith_lateral', group='gingiva', label='لثة الرباعية أنزل من لثة الثنية', min=0.0, max=1.5, unit='mm', needs='calib', enabled=True),
    dict(id='zenith_canine', group='gingiva', label='لثة الناب بمستوى لثة الثنية', min=-0.5, max=0.5, unit='mm', needs='calib', enabled=True),
    dict(id='zenith_sym', group='gingiva', label='تماثل مستوى لثة الثنيتين', max=0.5, unit='mm', needs='calib', enabled=True),
    dict(id='incisal_step', group='teeth', label='الحافة القاطعة للرباعية أقصر من الثنية', min=0.5, max=1.5, unit='mm', needs='calib', enabled=True),
    dict(id='midline_dev', group='face', label='انحراف خط الوسط السني عن خط الوجه', max=2.0, unit='mm', needs='face', enabled=True),
    dict(id='commissure_cant', group='face', label='ميل خط زوايا الفم عن خط العينين', max=2.0, unit='deg', needs='face', enabled=True),
    dict(id='incisal_cant', group='face', label='ميل مستوى الأنياب (المستوى القاطع) عن خط العينين', max=2.0, unit='deg', needs='face', enabled=True),
    dict(id='gingival_display', group='face', label='ظهور اللثة عند الابتسامة', min=0.0, max=2.0, unit='mm', needs='face', enabled=True),
    dict(id='buccal_corridor', group='face', label='عرض الأسنان الظاهرة إلى عرض الابتسامة', min=0.82, max=0.92, unit='ratio', needs='face', enabled=True),
    dict(id='smile_arc', group='face', label='قوس الابتسامة يوازي الشفة السفلى', unit='class', needs='face', enabled=True),
    dict(id='rest_display', group='face', label='ظهور الثنية والشفاه في الراحة', min=1.0, max=3.0, target=2.0, unit='mm', needs='rest', enabled=True),
    dict(id='alar_cant', group='face', label='ميل خط جناحي الأنف عن خط العينين', max=2.0, unit='deg', needs='face', enabled=True),
    dict(id='canine_alar', group='face', label='عرض ما بين الأنياب إلى عرض قاعدة الأنف (دليل تقليدي)', min=0.85, max=1.15, unit='ratio', needs='face', enabled=False),
    dict(id='lower_third', group='face', label='الثلث السفلي إلى الأوسط من الوجه', min=0.90, max=1.10, unit='ratio', needs='face', enabled=True),
]

SHADES = {
    'BL1': '#F7F4EE', 'BL2': '#F3EFE6', 'BL3': '#EFEADF', 'BL4': '#EBE5D8',
    'A1': '#EEE6D4', 'A2': '#E9DDC4', 'A3': '#E2D2B5', 'A3.5': '#DCC8A7', 'A4': '#D2BB97',
    'B1': '#F0E9D8', 'B2': '#EAE0C8', 'B3': '#E1D2B1', 'C1': '#E5DDCC', 'C2': '#DCD1BC', 'D2': '#E6DCCB',
}

SMILE_ARC_LABEL = {'consonant': 'متوافق مع الشفة', 'flat': 'مسطّح', 'reverse': 'معكوس', 'nonparallel': 'غير موازٍ للشفة'}

IRIS_MM = 11.7            # mean horizontal visible iris diameter in adults
IDEAL_W = {'13': 0.49, '12': 0.7, '11': 1, '21': 1, '22': 0.7, '23': 0.49}
IDEAL_L = {'13': 0.95, '12': 0.85, '11': 1, '21': 1, '22': 0.85, '23': 0.95}
