// Every response type is generated from contracts/openapi.json (npm run gen:types); never written by hand.
import type { components } from "./schema";

type C = components["schemas"];

export type Health = C["Health"];
export type FeatureList = C["FeatureList"];
export type FeatureListItem = C["FeatureListItem"];
export type FeatureDetail = C["FeatureDetail"];
export type Example = C["Example"];
export type LogitEntry = C["LogitEntry"];
export type ErrorBody = C["ErrorBody"];
export type LabelStatus = FeatureListItem["label_status"];

export type InspectRequest = C["InspectRequest"];
export type InspectResponse = C["InspectResponse"];
export type InspectToken = C["InspectToken"];
export type TokenFeature = C["TokenFeature"];
export type SteerRequest = C["SteerRequest"];
export type SteerResponse = C["SteerResponse"];
export type SteerMode = SteerRequest["mode"];
export type SteeringExample = C["SteeringExample"];
export type SteeringExampleList = C["SteeringExampleList"];
export type Run = C["Run"];
export type RunList = C["RunList"];
export type MetricsResponse = C["MetricsResponse"];
export type CurvePoint = C["CurvePoint"];
export type DensityBin = C["DensityBin"];
