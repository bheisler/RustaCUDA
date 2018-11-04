#[macro_use]
extern crate quote;
extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;

use proc_macro2::{Ident, Span, TokenStream};
use syn::{
    parse_str, Data, DataEnum, DataStruct, DataUnion, DeriveInput, Field, Fields, Generics,
    TypeParamBound,
};

use proc_macro::TokenStream as BaseTokenStream;

#[proc_macro_derive(DeviceCopy)]
pub fn derive_device_copy(input: BaseTokenStream) -> BaseTokenStream {
    let ast = syn::parse(input).unwrap();
    let gen = impl_device_copy(&ast);
    BaseTokenStream::from(gen)
}

fn impl_device_copy(input: &DeriveInput) -> TokenStream {
    let input_type = &input.ident;

    // Generate the code to check all fields of the derived struct
    let check_types_code = match input.data {
        Data::Struct(ref data_struct) => type_check_struct(data_struct),
        Data::Enum(ref data_enum) => type_check_enum(data_enum),
        Data::Union(ref data_union) => type_check_union(data_union),
    };

    let type_test_func_name = format!(
        "__verify_{}_can_implement_DeviceCopy",
        input_type.to_string()
    );
    let type_test_func_ident = Ident::new(&type_test_func_name, Span::call_site());

    let generics = add_bound_to_generics(&input.generics);
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    let generated_code = quote!{
        unsafe impl#impl_generics ::rustacuda::memory::DeviceCopy for #input_type#type_generics #where_clause {}

        #[doc(hidden)]
        #[allow(all)]
        fn #type_test_func_ident#impl_generics(value: &#input_type#type_generics) #where_clause {
            #check_types_code
        }
    };

    TokenStream::from(generated_code)
}

fn add_bound_to_generics(generics: &Generics) -> Generics {
    let mut new_generics = generics.clone();
    let bound: TypeParamBound =
        parse_str(&quote!{::rustacuda::memory::DeviceCopy}.to_string()).unwrap();

    for type_param in &mut new_generics.type_params_mut() {
        type_param.bounds.push(bound.clone())
    }

    new_generics
}

fn type_check_struct(s: &DataStruct) -> TokenStream {
    let checks = match s.fields {
        Fields::Named(ref named_fields) => {
            let fields: Vec<&Field> = named_fields.named.iter().collect();
            check_fields(&fields)
        }
        Fields::Unnamed(ref unnamed_fields) => {
            let fields: Vec<&Field> = unnamed_fields.unnamed.iter().collect();
            check_fields(&fields)
        }
        Fields::Unit => vec![],
    };
    quote!(
        #(#checks)*
    )
}

fn type_check_enum(s: &DataEnum) -> TokenStream {
    let mut checks = vec![];

    for variant in &s.variants {
        match variant.fields {
            Fields::Named(ref named_fields) => {
                let fields: Vec<&Field> = named_fields.named.iter().collect();
                checks.extend(check_fields(&fields));
            }
            Fields::Unnamed(ref unnamed_fields) => {
                let fields: Vec<&Field> = unnamed_fields.unnamed.iter().collect();
                checks.extend(check_fields(&fields));
            }
            Fields::Unit => {}
        }
    }
    quote!(
        #(#checks)*
    )
}

fn type_check_union(s: &DataUnion) -> TokenStream {
    let fields: Vec<&Field> = s.fields.named.iter().collect();
    let checks = check_fields(&fields);
    quote!(
        #(#checks)*
    )
}

fn check_fields(fields: &Vec<&Field>) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let field_type = &field.ty;
            quote!{
                {
                    fn assert_impl<T: ::rustacuda::memory::DeviceCopy>() {}
                    assert_impl::<#field_type>();
                }
            }
        }).collect()
}
