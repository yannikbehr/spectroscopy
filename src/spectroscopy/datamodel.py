import numpy as np
import datetime
from spectroscopy.class_factory import _base_class_factory, _buffer_class_factory


__Instrument = _base_class_factory('__Instrument', 'base',
	class_properties=[
		('tags',(set,)),
		('name',(np.str_,)),
		('sensor_id',(np.str_,)),
		('location',(np.str_,)),
		('no_bits',(np.int_,)),
		('type',(np.str_,)),
		('description',(np.str_,))],
	class_references=[])


__InstrumentBuffer = _buffer_class_factory(
	'__InstrumentBuffer', 
	class_properties=[
		('tags',(set,)),
		('name',(np.str_,)),
		('sensor_id',(np.str_,)),
		('location',(np.str_,)),
		('no_bits',(np.int_,)),
		('type',(np.str_,)),
		('description',(np.str_,))],
	class_references=[])


class InstrumentBuffer(__InstrumentBuffer):
	'''
	Description of the spectrometer.

	:type tags: set
	:param tags: List of human readable tags
	:type name: str
	:param name: Station name
	:type sensor_id: str
	:param sensor_id: Serial number
	:type location: str
	:param location: Name of sensor location
	:type no_bits: int
	:param no_bits: The number of bits used by the analog-to-digital converter.

	:type type: str
	:param type: The spectrometer type (e.g. DOAS, FlySpec, etc.).
	:type description: str
	:param description: Any additional information on the instrument that may be relevant.
	'''

class _Instrument(__Instrument):
	'''
	'''


__Target = _base_class_factory('__Target', 'base',
	class_properties=[
		('tags',(set,)),
		('target_id',(np.str_,)),
		('name',(np.str_,)),
		('position',(np.ndarray, np.float_)),
		('position_error',(np.ndarray, np.float_)),
		('description',(np.str_,))],
	class_references=[])


__TargetBuffer = _buffer_class_factory(
	'__TargetBuffer', 
	class_properties=[
		('tags',(set,)),
		('target_id',(np.str_,)),
		('name',(np.str_,)),
		('position',(np.ndarray, np.float_)),
		('position_error',(np.ndarray, np.float_)),
		('description',(np.str_,))],
	class_references=[])


class TargetBuffer(__TargetBuffer):
	'''
	Description of the plume location.

	:type tags: set
	:param tags: List of human readable tags
	:type target_id: str
	:param target_id: A unique, human readable ID
	:type name: str
	:param name: Descriptive name
	:type position: :class:`numpy.ndarray`
	:param position: Position of a plume in decimal degrees for longitude and latitude and m above sea level for elevation (lon, lat, elev).
	:type position_error: :class:`numpy.ndarray`
	:param position_error: Errors [degrees,degrees, m]
	:type description: str
	:param description: Any additional information on the plume that may be relevant.

	'''

class _Target(__Target):
	'''
	'''


__RawDataType = _base_class_factory('__RawDataType', 'base',
	class_properties=[
		('tags',(set,)),
		('d_var_unit',(np.str_,)),
		('ind_var_unit',(np.str_,)),
		('name',(np.str_,)),
		('acquisition',(np.str_,))],
	class_references=[])


__RawDataTypeBuffer = _buffer_class_factory(
	'__RawDataTypeBuffer', 
	class_properties=[
		('tags',(set,)),
		('d_var_unit',(np.str_,)),
		('ind_var_unit',(np.str_,)),
		('name',(np.str_,)),
		('acquisition',(np.str_,))],
	class_references=[])


class RawDataTypeBuffer(__RawDataTypeBuffer):
	'''
	

	:type tags: set
	:param tags: List of human readable tags
	:type d_var_unit: str
	:param d_var_unit: Unit of dependent variable
	:type ind_var_unit: str
	:param ind_var_unit: Unit of independent variable
	:type name: str
	:param name: Descriptive name (e.g. dark, offset, measurement, raw retrieval)
	:type acquisition: str
	:param acquisition: The type of acquisition (e.g. mobile, stationary)
	'''

class _RawDataType(__RawDataType):
	'''
	'''


__DataQualityType = _base_class_factory('__DataQualityType', 'base',
	class_properties=[
		('tags',(set,)),
		('name',(np.str_,)),
		('reference',(np.str_,))],
	class_references=[])


__DataQualityTypeBuffer = _buffer_class_factory(
	'__DataQualityTypeBuffer', 
	class_properties=[
		('tags',(set,)),
		('name',(np.str_,)),
		('reference',(np.str_,))],
	class_references=[])


class DataQualityTypeBuffer(__DataQualityTypeBuffer):
	'''
	A data quality description.

	:type tags: set
	:param tags: List of human readable IDs
	:type name: str
	:param name: Descriptive name (e.g. Intensity of the light source)
	:type reference: str
	:param reference: Reference to more detailed description
	'''

class _DataQualityType(__DataQualityType):
	'''
	'''


__RawData = _base_class_factory('__RawData', 'extendable',
	class_properties=[
		('tags',(set,)),
		('inc_angle',(np.ndarray, np.float_)),
		('inc_angle_error',(np.ndarray, np.float_)),
		('bearing',(np.ndarray, np.float_)),
		('bearing_error',(np.ndarray, np.float_)),
		('position',(np.ndarray, np.float_)),
		('position_error',(np.ndarray, np.float_)),
		('path_length',(np.ndarray, np.float_)),
		('path_length_error',(np.ndarray, np.float_)),
		('d_var',(np.ndarray, np.float_)),
		('ind_var',(np.ndarray, np.float_)),
		('datetime',(np.ndarray, datetime.datetime)),
		('data_quality',(np.ndarray, np.float_)),
		('integration_time',(np.ndarray, np.float_)),
		('no_averages',(np.float_,)),
		('temperature',(np.float_,)),
		('user_notes',(np.str_,))],
	class_references=[
		('instrument',(_Instrument,)),
		('target',(_Target,)),
		('type',(_RawDataType,)),
		('data_quality_type',(np.ndarray, _DataQualityType))])


__RawDataBuffer = _buffer_class_factory(
	'__RawDataBuffer', 
	class_properties=[
		('tags',(set,)),
		('inc_angle',(np.ndarray, np.float_)),
		('inc_angle_error',(np.ndarray, np.float_)),
		('bearing',(np.ndarray, np.float_)),
		('bearing_error',(np.ndarray, np.float_)),
		('position',(np.ndarray, np.float_)),
		('position_error',(np.ndarray, np.float_)),
		('path_length',(np.ndarray, np.float_)),
		('path_length_error',(np.ndarray, np.float_)),
		('d_var',(np.ndarray, np.float_)),
		('ind_var',(np.ndarray, np.float_)),
		('datetime',(np.ndarray, datetime.datetime)),
		('data_quality',(np.ndarray, np.float_)),
		('integration_time',(np.ndarray, np.float_)),
		('no_averages',(np.float_,)),
		('temperature',(np.float_,)),
		('user_notes',(np.str_,))],
	class_references=[
		('instrument',(_Instrument,)),
		('target',(_Target,)),
		('type',(_RawDataType,)),
		('data_quality_type',(np.ndarray, _DataQualityType))])


class RawDataBuffer(__RawDataBuffer):
	'''
	Raw data is data that requires further processing to become meaningful for gaschemistry analysis. It can be, for example, spectra recorded by a spectrometer or unscaled concentration measured by a laser diode. If an instrument provides pre-processed data, this element may only hold meta information about the raw data but not the raw data itself (e.g. electro-chemical sensors). 

	:type tags: set
	:param tags: List of human readable tags
	:type instrument: reference to Instrument
	:param instrument: Reference to the instrument recording the data.
	:type target: reference to Target
	:param target: Reference to the target plume.
	:type type: reference to RawDataType
	:param type: Reference to the raw-data-type.
	:type inc_angle: :class:`numpy.ndarray`
	:param inc_angle: Inclinitation of measurement direction from vertical. For a transect all angles would typically be the same, e.g. 0.0 if the spectrometer was pointing up.

	:type inc_angle_error: :class:`numpy.ndarray`
	:param inc_angle_error: Uncertainty of inclination angle in degrees.
	:type bearing: :class:`numpy.ndarray`
	:param bearing: Bearing of the scan plane in degrees from grid north.
	:type bearing_error: :class:`numpy.ndarray`
	:param bearing_error: Scan bearing uncertainty.
	:type position: :class:`numpy.ndarray`
	:param position: The position of the spectrometer in decimal longitude, latitude, and elevation in m above sea level  (lon, lat, elev).
	:type position_error: :class:`numpy.ndarray`
	:param position_error: Instrument location uncertainty.
	:type path_length: :class:`numpy.ndarray`
	:param path_length: Path length of a scan [m].
	:type path_length_error: :class:`numpy.ndarray`
	:param path_length_error: Path length uncertainty.
	:type d_var: :class:`numpy.ndarray`
	:param d_var: Dependent variable e.g. measured spectra, concentration
	:type ind_var: :class:`numpy.ndarray`
	:param ind_var: Independent variable e.g. wavelengths, time
	:type datetime: :class:`numpy.ndarray`
	:param datetime: Date Time of recording in ISO 8601 format.
	:type data_quality: :class:`numpy.ndarray`
	:param data_quality: Data quality parameters.
	:type data_quality_type: :class:`numpy.ndarray`
	:param data_quality_type: List of references to data-quality-type.
	:type integration_time: :class:`numpy.ndarray`
	:param integration_time: Exposure time [s].
	:type no_averages: float
	:param no_averages: Number/time measurements are averaged over.
	:type temperature: float
	:param temperature: Temperature at the site or in the instrument [degC].
	:type user_notes: str
	:param user_notes: Any additional information relevant to the measurements.
	'''

class _RawData(__RawData):
	'''
	'''


__Method = _base_class_factory('__Method', 'extendable',
	class_properties=[
		('tags',(set,)),
		('name',(np.str_,)),
		('description',(np.str_,)),
		('settings',(np.str_,)),
		('reference',(np.str_,))],
	class_references=[])


__MethodBuffer = _buffer_class_factory(
	'__MethodBuffer', 
	class_properties=[
		('tags',(set,)),
		('name',(np.str_,)),
		('description',(np.str_,)),
		('settings',(np.str_,)),
		('reference',(np.str_,))],
	class_references=[])


class MethodBuffer(__MethodBuffer):
	'''
	Desription of analysis methods.

	:type tags: set
	:param tags: List of human readable tags
	:type name: str
	:param name: Name of software/method
	:type description: str
	:param description: Short method summary
	:type settings: json
	:param settings: Settings/setup relevant to reproduce results in JSON format.
	:type reference: str
	:param reference: URI  to more detailed method description.
	'''

class _Method(__Method):
	'''
	'''


__GasFlow = _base_class_factory('__GasFlow', 'base',
	class_properties=[
		('tags',(set,)),
		('vx',(np.ndarray, np.float_)),
		('vx_error',(np.ndarray, np.float_)),
		('vy',(np.ndarray, np.float_)),
		('vy_error',(np.ndarray, np.float_)),
		('vz',(np.ndarray, np.float_)),
		('vz_error',(np.ndarray, np.float_)),
		('unit',(np.str_,)),
		('position',(np.ndarray, np.float_)),
		('position_error',(np.ndarray, np.float_)),
		('grid_bearing',(np.float_,)),
		('grid_increments',(np.ndarray, np.float_)),
		('pressure',(np.float_,)),
		('temperature',(np.float_,)),
		('datetime',(np.ndarray, datetime.datetime)),
		('user_notes',(np.str_,))],
	class_references=[
		('methods',(np.ndarray, _Method))])


__GasFlowBuffer = _buffer_class_factory(
	'__GasFlowBuffer', 
	class_properties=[
		('tags',(set,)),
		('vx',(np.ndarray, np.float_)),
		('vx_error',(np.ndarray, np.float_)),
		('vy',(np.ndarray, np.float_)),
		('vy_error',(np.ndarray, np.float_)),
		('vz',(np.ndarray, np.float_)),
		('vz_error',(np.ndarray, np.float_)),
		('unit',(np.str_,)),
		('position',(np.ndarray, np.float_)),
		('position_error',(np.ndarray, np.float_)),
		('grid_bearing',(np.float_,)),
		('grid_increments',(np.ndarray, np.float_)),
		('pressure',(np.float_,)),
		('temperature',(np.float_,)),
		('datetime',(np.ndarray, datetime.datetime)),
		('user_notes',(np.str_,))],
	class_references=[
		('methods',(np.ndarray, _Method))])


class GasFlowBuffer(__GasFlowBuffer):
	'''
	Can either contain estimates of plume velocity or wind speed. Plume velocity estimates can either come from direct measurements (e.g. image processing) or meteorological observations (e.g. wind speed). Wind speed estimates can be either from direct measurements (e.g. weather stations) or meteorological models. Both, plume velocity and wind speed can be either described on a regular 4D local cartesian grid or at a single point. Grids are assumed to be right-handed Cartesian coordinate systems with uniform grid point spacing along any direction.

	:type tags: set
	:param tags: List of human readable tags
	:type methods: :class:`numpy.ndarray`
	:param methods: List of references to methods used to compute gas flow
	:type vx: :class:`numpy.ndarray`
	:param vx: x component of gas flow vector (wrt local grid or east) 
	:type vx_error: :class:`numpy.ndarray`
	:param vx_error: x component error
	:type vy: :class:`numpy.ndarray`
	:param vy: y component of gas flow vector (wrt local grid or east)
	:type vy_error: :class:`numpy.ndarray`
	:param vy_error: y component error
	:type vz: :class:`numpy.ndarray`
	:param vz: z component of gas flow vector (wrt local grid or east)
	:type vz_error: :class:`numpy.ndarray`
	:param vz_error: z component error
	:type unit: str
	:param unit: Physical unit of gas flow vector
	:type position: :class:`numpy.ndarray`
	:param position: Measurement location or grid origin in decimal degrees for longitude and latitude and m above sea level for elevation (lon, lat, elev).
	:type position_error: :class:`numpy.ndarray`
	:param position_error: Location uncertainty
	:type grid_bearing: float
	:param grid_bearing: X-axis angle from grid north in decimal degrees.
	:type grid_increments: :class:`numpy.ndarray`
	:param grid_increments: Grid increments along the x-, y-, and z-axis 
	:type pressure: float
	:param pressure: Atmospheric pressure [Pa]
	:type temperature: float
	:param temperature: Temperature [degC]
	:type datetime: :class:`numpy.ndarray`
	:param datetime: Date Time [UTC] in ISO 8601 format
	:type user_notes: str
	:param user_notes: Any additional information that may be relevant.
	'''

class _GasFlow(__GasFlow):
	'''
	'''


__Concentration = _base_class_factory('__Concentration', 'extendable',
	class_properties=[
		('tags',(set,)),
		('rawdata_indices',(np.ndarray, np.int_)),
		('gas_species',(np.str_,)),
		('value',(np.ndarray, np.float_)),
		('value_error',(np.ndarray, np.float_)),
		('unit',(np.str_,)),
		('datetime',(np.ndarray, datetime.datetime)),
		('analyst_contact',(np.str_,)),
		('user_notes',(np.str_,))],
	class_references=[
		('method',(_Method,)),
		('gasflow',(_GasFlow,)),
		('rawdata',(np.ndarray, _RawData))])


__ConcentrationBuffer = _buffer_class_factory(
	'__ConcentrationBuffer', 
	class_properties=[
		('tags',(set,)),
		('rawdata_indices',(np.ndarray, np.int_)),
		('gas_species',(np.str_,)),
		('value',(np.ndarray, np.float_)),
		('value_error',(np.ndarray, np.float_)),
		('unit',(np.str_,)),
		('datetime',(np.ndarray, datetime.datetime)),
		('analyst_contact',(np.str_,)),
		('user_notes',(np.str_,))],
	class_references=[
		('method',(_Method,)),
		('gasflow',(_GasFlow,)),
		('rawdata',(np.ndarray, _RawData))])


class ConcentrationBuffer(__ConcentrationBuffer):
	'''
	Describes different types of gas concentration such as path concentration as inferred from spectrometers or volumetric concentration as measured by electro-chemical sensors.

	:type tags: set
	:param tags: List of human readable tags
	:type method: reference to Method
	:param method: Reference to the method used to obtain concentration estimates.
	:type gasflow: reference to GasFlow
	:param gasflow: Reference to gas flow model (if applicable).
	:type rawdata: :class:`numpy.ndarray`
	:param rawdata: Reference to raw measurements (if applicable).
	:type rawdata_indices: :class:`numpy.ndarray`
	:param rawdata_indices: Index of raw data used to estimate concentration
	:type gas_species: str
	:param gas_species: Gas type (e.g. SO2)
	:type value: :class:`numpy.ndarray`
	:param value: Concentration estimate.
	:type value_error: :class:`numpy.ndarray`
	:param value_error: Concentration uncertainty.
	:type unit: str
	:param unit: Unit of gas concentration.
	:type datetime: :class:`numpy.ndarray`
	:param datetime: Dates of concentration estimates in ISO 8601 format
	:type analyst_contact: str
	:param analyst_contact: Contact (e.g. email) of person running software
	:type user_notes: str
	:param user_notes: Any additional information that may be relevant.
	'''

class _Concentration(__Concentration):
	'''
	'''


__Flux = _base_class_factory('__Flux', 'base',
	class_properties=[
		('tags',(set,)),
		('concentration_indices',(np.ndarray, np.int_)),
		('value',(np.ndarray, np.float_)),
		('value_error',(np.ndarray, np.float_)),
		('datetime',(np.ndarray, datetime.datetime)),
		('unit',(np.str_,)),
		('analyst_contact',(np.str_,)),
		('user_notes',(np.str_,))],
	class_references=[
		('method',(_Method,)),
		('concentration',(_Concentration,)),
		('gasflow',(_GasFlow,))])


__FluxBuffer = _buffer_class_factory(
	'__FluxBuffer', 
	class_properties=[
		('tags',(set,)),
		('concentration_indices',(np.ndarray, np.int_)),
		('value',(np.ndarray, np.float_)),
		('value_error',(np.ndarray, np.float_)),
		('datetime',(np.ndarray, datetime.datetime)),
		('unit',(np.str_,)),
		('analyst_contact',(np.str_,)),
		('user_notes',(np.str_,))],
	class_references=[
		('method',(_Method,)),
		('concentration',(_Concentration,)),
		('gasflow',(_GasFlow,))])


class FluxBuffer(__FluxBuffer):
	'''
	Flux estimates based on concentration estimates and a gas flow model.

	:type tags: set
	:param tags: List of human readable tags
	:type method: reference to Method
	:param method: Reference to software used
	:type concentration: reference to Concentration
	:param concentration: Reference to concentration values used to compute flux.
	:type concentration_indices: :class:`numpy.ndarray`
	:param concentration_indices: Index of concentrations used to compute flux
	:type gasflow: reference to GasFlow
	:param gasflow: 
	:type value: :class:`numpy.ndarray`
	:param value: Flux estimates
	:type value_error: :class:`numpy.ndarray`
	:param value_error: Flux estimate errors
	:type datetime: :class:`numpy.ndarray`
	:param datetime: Datetime (in ISO 8601 format) that is assigned to the flux estimates.
	:type unit: str
	:param unit: Physical unit of flux.
	:type analyst_contact: str
	:param analyst_contact: Contact (e.g. email) of person running software
	:type user_notes: str
	:param user_notes: Any additional information that may be relevant.
	'''

class _Flux(__Flux):
	'''
	'''


__PreferredFlux = _base_class_factory('__PreferredFlux', 'base',
	class_properties=[
		('tags',(set,)),
		('flux_indices',(np.ndarray, np.int_)),
		('datetime',(np.ndarray, datetime.datetime)),
		('value',(np.ndarray, np.float_)),
		('value_error',(np.ndarray, np.float_)),
		('user_notes',(np.str_,))],
	class_references=[
		('fluxes',(np.ndarray, _Flux)),
		('method_id',(_Method,))])


__PreferredFluxBuffer = _buffer_class_factory(
	'__PreferredFluxBuffer', 
	class_properties=[
		('tags',(set,)),
		('flux_indices',(np.ndarray, np.int_)),
		('datetime',(np.ndarray, datetime.datetime)),
		('value',(np.ndarray, np.float_)),
		('value_error',(np.ndarray, np.float_)),
		('user_notes',(np.str_,))],
	class_references=[
		('fluxes',(np.ndarray, _Flux)),
		('method_id',(_Method,))])


class PreferredFluxBuffer(__PreferredFluxBuffer):
	'''
	Derived flux values either by selecting a subset or by integrating/averaging. These values would overlap with what is currently stored in FITS.

	:type tags: set
	:param tags: List of human readable tags
	:type fluxes: :class:`numpy.ndarray`
	:param fluxes: References to flux estimates used to compute derived flux
	:type flux_indices: :class:`numpy.ndarray`
	:param flux_indices: Indices of flux values used to compute derived flux
	:type datetime: :class:`numpy.ndarray`
	:param datetime: Dates of derived flux values in ISO 8601 format.
	:type value: :class:`numpy.ndarray`
	:param value: Derived flux value
	:type value_error: :class:`numpy.ndarray`
	:param value_error: Flux value error
	:type method_id: reference to Method
	:param method_id: Reference to method
	:type user_notes: str
	:param user_notes: Comments relevant for reproducing preferred fluxes
	'''

class _PreferredFlux(__PreferredFlux):
	'''
	'''


all_classes = [_Instrument, _Target, _RawDataType, _DataQualityType, _RawData, _Method, _GasFlow, _Concentration, _Flux, _PreferredFlux]
