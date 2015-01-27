#ifndef COMPONENT_DOT_H
#define COMPONENT_DOT_H

#include <SgtCore/Properties.h>

#include <ostream>
#include <string>

using std::string;

namespace SmartGridToolbox
{
   class ComponentInterface : virtual public HasPropertiesInterface
   {
      friend std::ostream& operator<<(std::ostream& os, const ComponentInterface& comp);

      public:
         virtual ~ComponentInterface() = default;
         virtual const std::string& id() const = 0;
         virtual const char* componentType() const = 0;

         template<typename T> T& as()
         {
            return dynamic_cast<T&>(*this);
         }

      protected:
         virtual void print(std::ostream& os) const = 0;
   };

   inline std::ostream& operator<<(std::ostream& os, const ComponentInterface& comp)
   {
      comp.print(os);
      return os;
   }

   class Component : virtual public ComponentInterface, public HasProperties
   {
      public:

      /// @name Lifecycle:
      /// @{
         
         Component(const std::string& id) : id_(id) {}
         virtual ~Component() = default;

      /// @}

      /// @name Component Type:
      /// @{
         
         virtual const char* componentType() const override {return "component";}

      /// @}
         
      /// @name Id:
      /// @{
         
         virtual const std::string& id() const override {return id_;}
         void setId(const std::string& id) {id_ = id;}
         SGT_PROP_GET_SET(id, Component, std::string, GetByConstRef, id, setId);

      /// @}
         
      protected:

      /// @name Printing:
      /// @{
         
         virtual void print(std::ostream& os) const
         {
            os << componentType() << ": " << id() << ":" << std::endl;
         }

      /// @}

      private:

         std::string id_ = "UNDEFINED";
   };
}

#endif // COMPONENT_DOT_H
