# SPL

## Camera and filter servers ##
Follow the instructions below
  - https://github.com/ArcetriAdaptiveOptics/plico_motor to start the tunable filter server
  - https://github.com/ArcetriAdaptiveOptics/pysilico to start the camera server

## SPL data acquisition and analysis ##
  - update parameters in spl/conf/SPL.conf
  - start python

from spl import SPL_controller as s
tt, piston = s.SPL_measurement_and_analysis()

  - Note down the saved TN:

Saved tracking number: 20220504_094318

  - Read the fringe matrix:

s.get_fringe_matrix('20220504_094318', display=True)

  - Individual images are saved in conf.measurement_path as FITS files: tn/image_xxx.fits

  - to access the camera and filter used by the above function (for debugging)

camera = s.define_camera()
filtro = s.define_filter()

  - if using a non-standard camera or filter, define them using plico_motor and pysilico and then:

tt, piston = s.SPL_measurement_and_analysis(camera, filter)

