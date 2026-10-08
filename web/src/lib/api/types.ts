// Every response type comes from generated files, never written by hand.
// schema.d.ts is generated from contracts/openapi.json. pending.d.ts is generated from
// server/app/models.py for endpoints that are not in the contract yet; once Mirdul routes
// them and the contract is regenerated, switch these imports to schema.d.ts.
import type { components as Contract } from "./schema";
import type { components as Pending } from "./pending";

type C = Contract["schemas"];
type P = Pending["schemas"];

export type Health = C["Health"];
export type FeatureList = C["FeatureList"];
export type FeatureListItem = C["FeatureListItem"];
export type FeatureDetail = C["FeatureDetail"];
export type Example = C["Example"];
export type LogitEntry = C["LogitEntry"];
export type ErrorBody = C["ErrorBody"];
export type LabelStatus = FeatureListItem["label_status"];

export type InspectRequest = P["InspectRequest"];
export type InspectResponse = P["InspectResponse"];
export type InspectToken = P["InspectToken"];
export type TokenFeature = P["TokenFeature"];
export type SteerRequest = P["SteerRequest"];
export type SteerResponse = P["SteerResponse"];
export type SteerMode = SteerRequest["mode"];
export type SteeringExample = P["SteeringExample"];
export type SteeringExampleList = P["SteeringExampleList"];
export type Run = P["Run"];
export type RunList = P["RunList"];
export type MetricsResponse = P["MetricsResponse"];
export type CurvePoint = P["CurvePoint"];
export type DensityBin = P["DensityBin"];
